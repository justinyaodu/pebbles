#ifndef ASSERT_GPU_H
#define ASSERT_GPU_H

#include <cassert>
#include <iostream>

#define gpuAssert(X) \
    do {                                                    \
        cudaError_t error = X;                              \
        if (error != cudaSuccess) {                         \
            std::cerr << #X                                 \
                    << ": " << cudaGetErrorName(error)      \
                    << ": " << cudaGetErrorString(error)    \
                    << std::endl;                           \
            std::exit(1);                                   \
        }                                                   \
    } while (0)

#endif
