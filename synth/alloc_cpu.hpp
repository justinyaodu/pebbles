#ifndef ALLOC_CPU_H
#define ALLOC_CPU_H

#include <cstdio>
#include <cstdlib>
#include <sys/mman.h>

// Allocate the specified number of bytes. Using this instead of malloc or
// new makes it possible to try huge pages and other tweaks.
void* alloc(size_t size) {
    void* ptr = mmap(
        nullptr,
        size,
        PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS,
        -1,
        0
    );

    if (ptr == MAP_FAILED) {
        std::perror(__func__);
        std::exit(1);
    }

    return ptr;
}

void dealloc(void* ptr, size_t size) {
    if (munmap(ptr, size)) {
        std::perror(__func__);
        std::exit(1);
    }
}

#endif
