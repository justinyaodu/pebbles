#ifndef BITSET_H
#define BITSET_H

#include <cassert>
#include <cstdint>

#include "alloc.hpp"

// a divided by b, rounded up.
#define CEIL_DIV(A, B) (((A) + (B) - 1) / (B))

class BaseBitset {
protected:
    const size_t size;
    uint8_t* const bytes;

    BaseBitset(const size_t size) :
        size(size),
        bytes((uint8_t*) alloc(CEIL_DIV(size, 8))) {}

    ~BaseBitset() {
        dealloc(bytes, CEIL_DIV(size, 8));
    }
};

class SingleThreadedBitset : BaseBitset {
public:
    SingleThreadedBitset(const size_t size) : BaseBitset(size) {}

    // Set the bit at the specified index,
    // and return the previous value of that bit.
    bool test_and_set(uint32_t index) {
        assert(index < size);

        uint8_t byte = bytes[index / 8];
        uint8_t mask = 1 << (index % 8);

        // Return 1 if bit is already set, to avoid unnecessary writes.
        if (byte & mask) {
            return 1;
        }

        bytes[index / 8] = byte | mask;

        // byte still contains the value right before the or operation.
        return (byte & mask) >> (index % 8);
    }
};

class ThreadSafeBitset : BaseBitset {
public:
    ThreadSafeBitset(const size_t size) : BaseBitset(size) {}

    // Atomically set the bit at the specified index,
    // and return the previous value of that bit.
    bool test_and_set(uint32_t index) {
        assert(index < size);

        // This fetch isn't atomic, but if the bit is already 1, we save an
        // unnecessary atomic operation.
        uint8_t byte = bytes[index / 8];
        uint8_t mask = 1 << (index % 8);
        if (byte & mask) {
            return 1;
        }

        // Bit is currently not set, but there would be a race condition if
        // we set the bit and return 0, because another thread might set this
        // bit at the same time. Use an atomic operation to prevent this.
        byte = __atomic_fetch_or(&bytes[index / 8], mask, __ATOMIC_SEQ_CST);

        // byte contains the value right before the or operation.
        return (byte & mask) >> (index % 8);
    }
};

#endif
