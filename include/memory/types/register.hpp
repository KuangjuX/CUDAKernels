#pragma once

#include <cstddef>

namespace cuda_kernels::memory::types {

template <typename T, size_t _outer_dim>
struct Register {
    using dtype = T;

    // Length in subtiles.
    static constexpr size_t outer_dim = _outer_dim;
    // Internal layout within a subtile. Either 1 or 2.
    // static constexpr size_t inner_dim = _inner_dim;

    // The actual register vector data.
    dtype data[outer_dim];

    DEVICE dtype& operator[](size_t idx) { return data[idx]; }
    DEVICE const dtype& operator[](size_t idx) const { return data[idx]; }
    DEVICE void print(int tid) {
        printf("tid: %d\n", tid);
        for (int i = 0; i < outer_dim; i++) {
            printf("%.2f %.2f ", data[i].x, data[i].y);
        }
        printf("\n");
    }
};

template <typename T2>
struct SubRegTile {
    using dtype = T2;

    static constexpr int tile_size = 16;
    static constexpr int rows = tile_size;                         // 16
    static constexpr int cols = tile_size;                         // 16
    static constexpr int num_elements = rows * cols;               // 256
    static constexpr int elements_per_thread = num_elements / 32;  // 8

    static constexpr int packed_per_threads = elements_per_thread / 2;  // 4
    static constexpr int registers_per_threads =
        packed_per_threads * sizeof(T2) / 4;  // 4 or 8

    T2 data[packed_per_threads];

    DEVICE T2& operator[](size_t idx) { return data[idx]; }
};

template <typename T2, size_t _height, size_t _width>
struct RegTile {
    using dtype = T2;

    static constexpr size_t height = _height;
    static constexpr size_t width = _width;
    static constexpr size_t rows = height * SubRegTile<dtype>::rows;
    static constexpr size_t cols = width * SubRegTile<dtype>::cols;
    static constexpr size_t tile_size = SubRegTile<dtype>::tile_size;
    static constexpr size_t num_elements =
        height * width * SubRegTile<dtype>::num_elements;
    static constexpr size_t elements_per_thread =
        SubRegTile<dtype>::elements_per_thread * height * width;
    static constexpr size_t packed_per_threads =
        SubRegTile<dtype>::packed_per_threads * height * width;

    SubRegTile<dtype> tiles[height][width];
};

}  // namespace cuda_kernels::memory::types