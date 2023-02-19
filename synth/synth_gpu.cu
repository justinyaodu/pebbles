#ifndef SYNTH_GPU_H
#define SYNTH_GPU_H

#include <cstdint>
#include <thrust/scan.h>

#include "bitset_gpu.cu"
#include "expr.hpp"
#include "spec.hpp"
#include "synth.hpp"

#define BLOCK_SIZE 1024

struct DevicePassState {
    // Number of terms in the bank.
    uint32_t num_terms;

    // Did we find a solution in this pass?
    uint32_t found_sol;

    // If so, what's its index in the bank?
    uint32_t sol_idx;

    DevicePassState(uint32_t num_terms) :
            num_terms(num_terms),
            found_sol(false),
            sol_idx(0) {}
};

// See inclusive_scan.py for details.
__device__ inline void inclusive_scan(uint32_t* values, uint32_t length) {
    uint32_t step, idx;
    for (step = 1, idx = threadIdx.x * 2 + 1;
            step < length;
            step *= 2, __syncthreads()) {
        if (idx < length) {
            values[idx] += values[idx - step];
            idx = idx * 2 + 1;
        }
    }
    for (step = length / 4, idx = threadIdx.x * (2 * step) + (3 * step) - 1;
            step > 0;
            step /= 2, idx /= 2, __syncthreads()) {
        if (idx < length) {
            values[idx] += values[idx - step];
        }
    }
}

__global__ void pass_variable(
    int32_t height,
    DevicePassState* __restrict__ state,
    uint32_t result_mask,
    GPUBitset __restrict__ seen,
    uint32_t sol_result,
    uint32_t* __restrict__ term_results,
    uint32_t* __restrict__ term_lefts,
    uint32_t num_vars,
    const uint32_t* __restrict__ var_values,
    const int32_t* __restrict__ var_heights
) {
    uint32_t var_idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    uint32_t is_new = 0;
    uint32_t var_value;
    if (var_idx < num_vars && var_heights[var_idx] == height) {
        var_value = result_mask & var_values[var_idx];
        if (!GPUBitset_test_and_set(seen, var_value)) {
            is_new = 1;
        }
    }

    __shared__ uint32_t new_count[BLOCK_SIZE];
    new_count[threadIdx.x] = is_new;
    __syncthreads();
    inclusive_scan(new_count, BLOCK_SIZE);

    __shared__ uint32_t bank_segment_start;
    if (threadIdx.x == 0) {
        bank_segment_start = atomicAdd(&state->num_terms, new_count[BLOCK_SIZE - 1]);
    }
    __syncthreads();

    if (is_new) {
        uint32_t bank_idx = bank_segment_start + new_count[threadIdx.x] - 1;
        term_results[bank_idx] = var_value;
        term_lefts[bank_idx] = var_idx;
        if (var_value == sol_result) {
            state->found_sol = true;
            state->sol_idx = bank_idx;
        }
    }
}

class Synthesizer : public AbstractSynthesizer {
private:
    GPUBitset seen;
    DevicePassState* device_pass_state;

public:
    Synthesizer(Spec spec) : AbstractSynthesizer(spec),
            seen(GPUBitset_new(max_distinct_terms)) {
        DevicePassState state(num_terms);
        gpuAssert(cudaMalloc(&device_pass_state, sizeof(DevicePassState)));
        gpuAssert(cudaMemcpy(device_pass_state, &state, sizeof(DevicePassState),
                cudaMemcpyHostToDevice));
    }

private:
    int64_t finish_pass() {
        cudaDeviceSynchronize();

        DevicePassState state(0);
        gpuAssert(cudaMemcpy(&state, device_pass_state, sizeof(DevicePassState),
                cudaMemcpyDeviceToHost));
        num_terms = state.num_terms;
        return state.found_sol ? state.sol_idx : -1LL;
    }

    int64_t pass_Variable(int32_t height) {
        size_t vars_size = spec.num_vars * sizeof(uint32_t);
        uint32_t* device_var_values;
        int32_t* device_var_heights;
        gpuAssert(cudaMalloc(&device_var_values, vars_size));
        gpuAssert(cudaMalloc(&device_var_heights, vars_size));
        gpuAssert(cudaMemcpy(device_var_values, &spec.var_values[0], vars_size,
                cudaMemcpyHostToDevice));
        gpuAssert(cudaMemcpy(device_var_heights, &spec.var_heights[0], vars_size,
                cudaMemcpyHostToDevice));

        dim3 dim_grid(CEIL_DIV(spec.num_vars, BLOCK_SIZE));
        dim3 dim_block(BLOCK_SIZE);
        pass_variable<<<dim_grid, dim_block>>>(
            height,
            device_pass_state,
            result_mask,
            seen,
            spec.sol_result,
            term_results,
            term_lefts,
            spec.num_vars,
            device_var_values,
            device_var_heights
        );
        return finish_pass();
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
