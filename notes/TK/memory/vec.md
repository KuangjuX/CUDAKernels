# ThunderKittens 中的内存访问实现

## Vectorize Load/Store

### Global to Reg

```cpp
/**
 * @brief Load data into a register vector from a source array in global memory.
 *
 * @tparam RV The register vector type.
 * @tparam U The data type of the source array.
 * @param[out] dst The destination register vector to load data into.
 * @param[in] src The source array in global memory to load data from.
 */
template<ducks::rv::all RV, typename U>
__device__ inline static void load(RV &dst, const U *src) {
    using T2 = RV::dtype;
    using U2 = base_types::packing<U>::packed_type;
    using T = base_types::packing<T2>::unpacked_type;
    
    int laneid = ::kittens::laneid();
    
    __syncwarp();
    if constexpr (dst.inner_dim == 2) {
        #pragma unroll
        for(auto w = 0; w < (dst.outer_dim+3)/4; w++) {
            int idx = w*64 + (laneid/4)*8 + 2*(laneid%4);
            int o_dim = w*4 + (laneid/4) / 2;
            int i_dim = (laneid/4) % 2;
            // this should be a maximally coalesced load.
            if(idx < dst.outer_dim*16)
                dst[o_dim][i_dim] = base_types::convertor<T2, U2>::convert(*(U2*)&src[idx]);
        }
        __syncwarp();
        // now we need to do a bunch of shuffle_sync's to make sure everyone has everything they need.
        #pragma unroll
        for(auto w = 0; w < dst.outer_dim; w++) {
            int leader = 8*(w%4) + (laneid%4); // repeats every 64 columns
            dst[w][0] = packed_shfl_sync(MASK_ALL, dst[w][0], leader);
            dst[w][1] = packed_shfl_sync(MASK_ALL, dst[w][1], leader+4);
        }
    }
    else {
        // really hoping https://stackoverflow.com/questions/15029765/is-coalescing-triggered-for-accessing-memory-in-reverse-order is still true
        // otherwise there will be some pain :/
        #pragma unroll
        for(auto w = 0; w < (dst.outer_dim+1)/2; w++) {
            int idx = w*32 + (laneid%4)*8 + (laneid/4);
            int o_dim = w*2 + (laneid%4) / 2;
            // this should be a maximally coalesced load.
            if(idx < dst.outer_dim*16) {
                T tmp = base_types::convertor<T, U>::convert(src[idx]);
                if(laneid%2==0) dst[o_dim][0].x = tmp;
                else dst[o_dim][0].y = tmp;
            }
        }
        __syncwarp();
        // now we need to do a bunch of shuffle_sync's to make sure everyone has everything they need.
        #pragma unroll
        for(auto w = 0; w < dst.outer_dim; w++) {
            int leader = (laneid/4)*4 + 2*(w%2); // repeats every 64 columns
            dst[w][0].x = __shfl_sync(MASK_ALL, dst[w][0].x, leader);
            dst[w][0].y = __shfl_sync(MASK_ALL, dst[w][0].y, leader+1);
        }
    }
}
```

这里 ThunerKittens 自己实现了向量化寄存器的数据结构：

```cpp
/**
 * @brief Register vector structure.
 *
 * @tparam _T The packed data type used for the vector elements.
 * @tparam _outer_dim The size of the tile, in units of TILE_DIM (16).
 * @tparam _inner_dim This controls the layout of the tile in terms of which axis it maps on the register tile layout.
 *
 * Register vectors are used to accumulate and map values across tiles. You can do computation
 * on them directly if you want, but they're not designed to be maximally efficient vectors
 * as they have substantial duplication and strange layouts to help them work efficiently with
 * the register layouts used by the tensor cores. ThunderKittens wants you working with tiles
 * where possible!
 */
template<typename _T, size_t _outer_dim, size_t _inner_dim=1>
struct rv {
    using identifier = ducks::rv::identifier; ///< Type identifier for the rv structure.
    using dtype = _T; ///< Data type of the vector elements.

    static constexpr int outer_dim = _outer_dim; ///< Length in subtiles.
    static constexpr int inner_dim = _inner_dim; ///< Internal layout within a subtile. Either 1 or 2.

    dtype data[outer_dim][inner_dim]; ///< The actual register vector data.

    __device__ inline       dtype* operator[](size_t idx)       { return &data[idx][0]; } ///< A wrapper for indexing into vector data.
    __device__ inline const dtype* operator[](size_t idx) const { return &data[idx][0]; } ///< A wrapper for indexing into vector data.
    __device__ inline       dtype& operator[](int2 outin)       { return data[outin.x][outin.y]; } ///< A wrapper for indexing into vector data.
    __device__ inline const dtype& operator[](int2 outin) const { return data[outin.x][outin.y]; } ///< A wrapper for indexing into vector data.
};
```

有外层维度和内层维度，内层维度只可以取 1 或者 2，暂时不知道有啥作用。优化手段是向量化加载 + 内存合并访问，在内层维度为 1 的情况下，使用 warp 进行协作式加载，加载过程中 4 个线程一组，每组隔 8 个加载一个数：

- w = 0, land_id = 0, 1, 2, 3, idx = 0, 8, 16, 24
- w = 1, land_id = 0, 1, 2, 3, idx = 32, 40, 48, 56
- w = 0, land_id = 4, 5, 6, 7, idx = 1, 9, 17, 25
- w = 1, land_id = 4, 5, 6, 7, idx = 33, 41, 49, 57

然后相邻的四个线程进行数据同步，warp 中相邻四个线程里的数据是一样的，不同 warp 间不共享，需要手动算偏移量。