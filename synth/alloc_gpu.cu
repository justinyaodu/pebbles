#ifndef ALLOC_GPU_H
#define ALLOC_GPU_H

#include "gpu_assert.cu"

void* alloc(size_t size) {
    void* ptr;
    gpuAssert(cudaMallocManaged(&ptr, size));
    return ptr;
}

void dealloc(void* ptr, size_t size) {
    gpuAssert(cudaFree(ptr));
}

#endif
