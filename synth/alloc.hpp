#ifndef ALLOC_H
#define ALLOC_H

#ifdef __CUDACC__
#include "alloc_gpu.cu"
#else
#include "alloc_cpu.hpp"
#endif

#endif
