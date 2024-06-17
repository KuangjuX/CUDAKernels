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
    // DEVICE dtype& operator[](int2 outin) { return data[outin.x][outin.y]; }
    // DEVICE const dtype& operator[](int2 outin) const {
    //     return data[outin.x][outin.y];
    // }
};

}  // namespace cuda_kernels::memory::types