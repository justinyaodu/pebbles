#ifndef BITSET_GPU_H
#define BITSET_GPU_H

#include "gpu_assert.cu"
#include "util.hpp"

typedef uint32_t* GPUBitset;

__host__ GPUBitset GPUBitset_new(const size_t size) {
    uint32_t* words;
    size_t byte_size = CEIL_DIV(size, 8);
    gpuAssert(cudaMalloc(&words, byte_size));
    gpuAssert(cudaMemset(words, 0, byte_size));
    return words;
}

__device__ bool GPUBitset_test_and_set(GPUBitset bitset, uint32_t index) {
    uint32_t word = bitset[index / 32];
    uint32_t mask = 1 << (index % 32);

    // If the bit is already set, return early without doing an atomic
    // operation. This provides a speedup of ~1.2x.
    if (word & mask) {
        return 1;
    }

    word = atomicOr(&bitset[index / 32], mask);
    return (word & mask) >> (index % 32);
}

__device__ bool GPUBitset_test(GPUBitset bitset, uint32_t index) {
    return (bitset[index / 32] >> (index % 32)) & 1;
}

#endif
