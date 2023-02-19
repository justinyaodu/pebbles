#ifndef SYNTH_GPU_H
#define SYNTH_GPU_H

#include <cstdint>
#include <thrust/scan.h>

#include "bitset_gpu.cu"
#include "expr.hpp"
#include "spec.hpp"
#include "synth.hpp"

class Synthesizer : public AbstractSynthesizer {
private:
    GPUBitset seen;

public:
    Synthesizer(Spec spec) : AbstractSynthesizer(spec),
            seen(GPUBitset_new(max_distinct_terms)) {}

    int64_t pass_Variable(int32_t height) {
        return NOT_FOUND;
    }

    int64_t pass_Not(int32_t height) {
        return NOT_FOUND;
    }

    int64_t pass_And(int32_t height) {
        return NOT_FOUND;
    }

    int64_t pass_Or(int32_t height) {
        return NOT_FOUND;
    }

    int64_t pass_Xor(int32_t height) {
        return NOT_FOUND;
    }
};

#endif
