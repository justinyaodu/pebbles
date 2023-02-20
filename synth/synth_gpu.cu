#ifndef SYNTH_GPU_H
#define SYNTH_GPU_H

#include <cstdint>
#include <iostream>

#include "bitset_gpu.cu"
#include "expr.hpp"
#include "spec.hpp"
#include "synth.hpp"

#define BLOCK_SIZE 1024
#define TILE_SIZE 32
#define MAX_GRID_DIM_Y 65535

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
__device__ inline void inclusive_scan_and_syncthreads(
    uint32_t* values,
    uint32_t length
) {
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

// Add new terms to the bank.
__device__ inline void add_binary_terms(
    DevicePassState* __restrict__ state,
    uint32_t sol_result,
    uint32_t* __restrict__ term_results,
    uint32_t* __restrict__ term_lefts,
    uint32_t* __restrict__ term_rights,
    uint32_t is_new,
    uint32_t result,
    uint32_t left,
    uint32_t right
) {
    __shared__ uint32_t new_count[BLOCK_SIZE];
    new_count[threadIdx.x] = is_new;
    __syncthreads();
    inclusive_scan_and_syncthreads(new_count, BLOCK_SIZE);

    uint32_t batch_size = new_count[BLOCK_SIZE - 1];
    if (batch_size == 0) {
        return;
    }

    __shared__ uint32_t batch_results[BLOCK_SIZE];
    __shared__ uint32_t batch_lefts[BLOCK_SIZE];
    __shared__ uint32_t batch_rights[BLOCK_SIZE];

    if (is_new) {
        uint32_t batch_idx = new_count[threadIdx.x] - 1;
        batch_results[batch_idx] = result;
        batch_lefts[batch_idx] = left;
        batch_rights[batch_idx] = right;
    }
    __syncthreads();

    __shared__ uint32_t bank_segment_start;
    if (threadIdx.x == 0) {
        bank_segment_start = atomicAdd(&state->num_terms, batch_size);
    }
    __syncthreads();

    if (threadIdx.x < batch_size) {
        uint32_t bank_idx = bank_segment_start + threadIdx.x;
        uint32_t batch_result = batch_results[threadIdx.x];
        term_results[bank_idx] = batch_result;
        term_lefts[bank_idx] = batch_lefts[threadIdx.x];
        term_rights[bank_idx] = batch_rights[threadIdx.x];

        if (batch_result == sol_result) {
            state->found_sol = true;
            state->sol_idx = bank_idx;
        }
    }
}

// This is the same code as add_binary_terms, minus the code for right children.
__device__ inline void add_unary_terms(
    DevicePassState* __restrict__ state,
    uint32_t sol_result,
    uint32_t* __restrict__ term_results,
    uint32_t* __restrict__ term_lefts,
    uint32_t is_new,
    uint32_t result,
    uint32_t left
) {
    __shared__ uint32_t new_count[BLOCK_SIZE];
    new_count[threadIdx.x] = is_new;
    __syncthreads();
    inclusive_scan_and_syncthreads(new_count, BLOCK_SIZE);

    uint32_t batch_size = new_count[BLOCK_SIZE - 1];
    if (batch_size == 0) {
        return;
    }

    __shared__ uint32_t batch_results[BLOCK_SIZE];
    __shared__ uint32_t batch_lefts[BLOCK_SIZE];

    if (is_new) {
        uint32_t batch_idx = new_count[threadIdx.x] - 1;
        batch_results[batch_idx] = result;
        batch_lefts[batch_idx] = left;
    }
    __syncthreads();

    __shared__ uint32_t bank_segment_start;
    if (threadIdx.x == 0) {
        bank_segment_start = atomicAdd(&state->num_terms, batch_size);
    }
    __syncthreads();

    if (threadIdx.x < batch_size) {
        uint32_t bank_idx = bank_segment_start + threadIdx.x;
        uint32_t batch_result = batch_results[threadIdx.x];
        term_results[bank_idx] = batch_result;
        term_lefts[bank_idx] = batch_lefts[threadIdx.x];

        if (batch_result == sol_result) {
            state->found_sol = true;
            state->sol_idx = bank_idx;
        }
    }
}

#define RETURN_IF_FOUND_SOL \
    do {                                    \
        __shared__ uint32_t found_sol;      \
        if (threadIdx.x == 0) {             \
            found_sol = state->found_sol;   \
        }                                   \
        __syncthreads();                    \
        if (found_sol) {                    \
            return;                         \
        }                                   \
    } while (0);


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
    RETURN_IF_FOUND_SOL;

    uint32_t var_idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    uint32_t is_new = 0;
    uint32_t var_value;
    if (var_idx < num_vars && var_heights[var_idx] == height) {
        var_value = result_mask & var_values[var_idx];
        if (!GPUBitset_test_and_set(seen, var_value)) {
            is_new = 1;
        }
    }

    add_unary_terms(state, sol_result, term_results, term_lefts,
            is_new, var_value, var_idx);
}

__global__ void pass_not(
    DevicePassState* __restrict__ state,
    uint32_t result_mask,
    GPUBitset __restrict__ seen,
    uint32_t sol_result,
    uint32_t* __restrict__ term_results,
    uint32_t* __restrict__ term_lefts,
    uint32_t all_lefts_start,
    uint32_t all_lefts_end
) {
    RETURN_IF_FOUND_SOL;

    uint32_t left = all_lefts_start + blockIdx.x * BLOCK_SIZE + threadIdx.x;

    uint32_t is_new = 0;
    uint32_t result;
    if (left < all_lefts_end) {
        result = result_mask & (~term_results[left]);
        if (!GPUBitset_test_and_set(seen, result)) {
            is_new = 1;
        }
    }

    add_unary_terms(state, sol_result, term_results, term_lefts,
            is_new, result, left);
}

__global__ void pass_and(
    DevicePassState* __restrict__ state,
    uint32_t result_mask,
    GPUBitset __restrict__ seen,
    uint32_t sol_result,
    uint32_t* __restrict__ term_results,
    uint32_t* __restrict__ term_lefts,
    uint32_t* __restrict__ term_rights,
    uint32_t k,
    uint32_t n,
    uint32_t all_lefts_end,
    uint32_t all_rights_start,
    uint32_t all_rights_end
) {
    RETURN_IF_FOUND_SOL;

    uint32_t lefts_tile = blockIdx.x;
    uint32_t rights_tile = n - 1 - blockIdx.y;
    if (lefts_tile > rights_tile) {
        lefts_tile = n - (lefts_tile - (k + 1)) - 1;
        rights_tile = n - (rights_tile - k) - 1;
    }

    uint32_t lefts_base = lefts_tile * TILE_SIZE;
    uint32_t rights_base = rights_tile * TILE_SIZE;

    // TODO: unclear if tiling actually improves performance here.
    __shared__ uint32_t term_results_left_tile[TILE_SIZE];
    __shared__ uint32_t term_results_right_tile[TILE_SIZE];
    if (threadIdx.x < TILE_SIZE) {
        term_results_left_tile[threadIdx.x] = term_results[lefts_base + threadIdx.x];
        term_results_right_tile[threadIdx.x] = term_results[rights_base + threadIdx.x];
    }
    __syncthreads();

    uint32_t lefts_offset = threadIdx.x / TILE_SIZE;
    uint32_t rights_offset = threadIdx.x % TILE_SIZE;
    uint32_t left = lefts_base + lefts_offset;
    uint32_t right = rights_base + rights_offset;

    uint32_t is_new = 0;
    uint32_t result;
    if (left < all_lefts_end && all_rights_start <= right && right < all_rights_end) {
        uint32_t left_result = term_results_left_tile[lefts_offset];
        uint32_t right_result = term_results_right_tile[rights_offset];
        result = result_mask & (left_result & right_result);
        // Experimentally, checking result != left_result && result != right_result
        // before checking the bitset doesn't improve performance.
        if (!GPUBitset_test_and_set(seen, result)) {
            is_new = 1;
        }
    }

    add_binary_terms(state, sol_result, term_results, term_lefts, term_rights,
            is_new, result, left, right);
}

__global__ void pass_or(
    DevicePassState* __restrict__ state,
    uint32_t result_mask,
    GPUBitset __restrict__ seen,
    uint32_t sol_result,
    uint32_t* __restrict__ term_results,
    uint32_t* __restrict__ term_lefts,
    uint32_t* __restrict__ term_rights,
    uint32_t k,
    uint32_t n,
    uint32_t all_lefts_end,
    uint32_t all_rights_start,
    uint32_t all_rights_end
) {
    RETURN_IF_FOUND_SOL;

    uint32_t lefts_tile = blockIdx.x;
    uint32_t rights_tile = n - 1 - blockIdx.y;
    if (lefts_tile > rights_tile) {
        lefts_tile = n - (lefts_tile - (k + 1)) - 1;
        rights_tile = n - (rights_tile - k) - 1;
    }

    uint32_t lefts_base = lefts_tile * TILE_SIZE;
    uint32_t rights_base = rights_tile * TILE_SIZE;

    __shared__ uint32_t term_results_left_tile[TILE_SIZE];
    __shared__ uint32_t term_results_right_tile[TILE_SIZE];
    if (threadIdx.x < TILE_SIZE) {
        term_results_left_tile[threadIdx.x] = term_results[lefts_base + threadIdx.x];
        term_results_right_tile[threadIdx.x] = term_results[rights_base + threadIdx.x];
    }
    __syncthreads();

    uint32_t lefts_offset = threadIdx.x / TILE_SIZE;
    uint32_t rights_offset = threadIdx.x % TILE_SIZE;
    uint32_t left = lefts_base + lefts_offset;
    uint32_t right = rights_base + rights_offset;

    uint32_t is_new = 0;
    uint32_t result;
    if (left < all_lefts_end && all_rights_start <= right && right < all_rights_end) {
        uint32_t left_result = term_results_left_tile[lefts_offset];
        uint32_t right_result = term_results_right_tile[rights_offset];
        result = result_mask & (left_result | right_result);
        if (!GPUBitset_test_and_set(seen, result)) {
            is_new = 1;
        }
    }

    add_binary_terms(state, sol_result, term_results, term_lefts, term_rights,
            is_new, result, left, right);
}

__global__ void pass_xor(
    DevicePassState* __restrict__ state,
    uint32_t result_mask,
    GPUBitset __restrict__ seen,
    uint32_t sol_result,
    uint32_t* __restrict__ term_results,
    uint32_t* __restrict__ term_lefts,
    uint32_t* __restrict__ term_rights,
    uint32_t k,
    uint32_t n,
    uint32_t all_lefts_end,
    uint32_t all_rights_start,
    uint32_t all_rights_end
) {
    RETURN_IF_FOUND_SOL;

    uint32_t lefts_tile = blockIdx.x;
    uint32_t rights_tile = n - 1 - blockIdx.y;
    if (lefts_tile > rights_tile) {
        lefts_tile = n - (lefts_tile - (k + 1)) - 1;
        rights_tile = n - (rights_tile - k) - 1;
    }

    uint32_t lefts_base = lefts_tile * TILE_SIZE;
    uint32_t rights_base = rights_tile * TILE_SIZE;

    __shared__ uint32_t term_results_left_tile[TILE_SIZE];
    __shared__ uint32_t term_results_right_tile[TILE_SIZE];
    if (threadIdx.x < TILE_SIZE) {
        term_results_left_tile[threadIdx.x] = term_results[lefts_base + threadIdx.x];
        term_results_right_tile[threadIdx.x] = term_results[rights_base + threadIdx.x];
    }
    __syncthreads();

    uint32_t lefts_offset = threadIdx.x / TILE_SIZE;
    uint32_t rights_offset = threadIdx.x % TILE_SIZE;
    uint32_t left = lefts_base + lefts_offset;
    uint32_t right = rights_base + rights_offset;

    uint32_t is_new = 0;
    uint32_t result;
    if (left < all_lefts_end && all_rights_start <= right && right < all_rights_end) {
        result = result_mask & (term_results_left_tile[lefts_offset] ^ term_results_right_tile[rights_offset]);
        if (!GPUBitset_test_and_set(seen, result)) {
            is_new = 1;
        }
    }

    add_binary_terms(state, sol_result, term_results, term_lefts, term_rights,
            is_new, result, left, right);
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

    ~Synthesizer() {
        gpuAssert(cudaFree(device_pass_state));
    }

private:

#define KERNEL_SYNC_IF_DONE_RETURN \
    do {                                    \
        cudaDeviceSynchronize();            \
        DevicePassState state(0);           \
        gpuAssert(cudaMemcpy(               \
                &state,                     \
                device_pass_state,          \
                sizeof(DevicePassState),    \
                cudaMemcpyDeviceToHost));   \
        num_terms = state.num_terms;        \
        if (state.found_sol) {              \
            return state.sol_idx;           \
        }                                   \
    } while(0);


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
        KERNEL_SYNC_IF_DONE_RETURN;
        return NOT_FOUND;
    }

    int64_t pass_Not(int32_t height) {
        int64_t all_lefts_start = terms_with_height_start(height - 1);
        int64_t all_lefts_end = terms_with_height_end(height - 1);

        dim3 dim_grid(CEIL_DIV(all_lefts_end - all_lefts_start, BLOCK_SIZE));
        dim3 dim_block(BLOCK_SIZE);
        pass_not<<<dim_grid, dim_block>>>(
            device_pass_state,
            result_mask,
            seen,
            spec.sol_result,
            term_results,
            term_lefts,
            all_lefts_start,
            all_lefts_end
        );
        KERNEL_SYNC_IF_DONE_RETURN;
        return NOT_FOUND;
    }

    int64_t pass_And(int32_t height) {
        int64_t all_lefts_end = terms_with_height_end(height - 1);

        int64_t all_rights_start = terms_with_height_start(height - 1);
        int64_t all_rights_end = all_lefts_end;

        int64_t max_n = CEIL_DIV(all_rights_end, TILE_SIZE);
        for (int64_t k = all_rights_start / TILE_SIZE; k < max_n; k += MAX_GRID_DIM_Y) {
            int64_t n = std::min(k + MAX_GRID_DIM_Y, max_n);

            dim3 dim_grid(CEIL_DIV((k + 1) + n, 2), n - k);
            dim3 dim_block(BLOCK_SIZE);

            pass_and<<<dim_grid, dim_block>>>(
                device_pass_state,
                result_mask,
                seen,
                spec.sol_result,
                term_results,
                term_lefts,
                term_rights,
                k,
                n,
                all_lefts_end,
                all_rights_start,
                all_rights_end
            );
            KERNEL_SYNC_IF_DONE_RETURN;
        }
        return NOT_FOUND;
    }

    int64_t pass_Or(int32_t height) {
        int64_t all_lefts_end = terms_with_height_end(height - 1);

        int64_t all_rights_start = terms_with_height_start(height - 1);
        int64_t all_rights_end = all_lefts_end;

        int64_t max_n = CEIL_DIV(all_rights_end, TILE_SIZE);
        for (int64_t k = all_rights_start / TILE_SIZE; k < max_n; k += MAX_GRID_DIM_Y) {
            int64_t n = std::min(k + MAX_GRID_DIM_Y, max_n);

            dim3 dim_grid(CEIL_DIV((k + 1) + n, 2), n - k);
            dim3 dim_block(BLOCK_SIZE);

            pass_or<<<dim_grid, dim_block>>>(
                device_pass_state,
                result_mask,
                seen,
                spec.sol_result,
                term_results,
                term_lefts,
                term_rights,
                k,
                n,
                all_lefts_end,
                all_rights_start,
                all_rights_end
            );
            KERNEL_SYNC_IF_DONE_RETURN;
        }
        return NOT_FOUND;
    }

    int64_t pass_Xor(int32_t height) {
        int64_t all_lefts_end = terms_with_height_end(height - 1);

        int64_t all_rights_start = terms_with_height_start(height - 1);
        int64_t all_rights_end = all_lefts_end;

        int64_t max_n = CEIL_DIV(all_rights_end, TILE_SIZE);
        for (int64_t k = all_rights_start / TILE_SIZE; k < max_n; k += MAX_GRID_DIM_Y) {
            int64_t n = std::min(k + MAX_GRID_DIM_Y, max_n);

            dim3 dim_grid(CEIL_DIV((k + 1) + n, 2), n - k);
            dim3 dim_block(BLOCK_SIZE);

            pass_xor<<<dim_grid, dim_block>>>(
                device_pass_state,
                result_mask,
                seen,
                spec.sol_result,
                term_results,
                term_lefts,
                term_rights,
                k,
                n,
                all_lefts_end,
                all_rights_start,
                all_rights_end
            );
            KERNEL_SYNC_IF_DONE_RETURN;
        }
        return NOT_FOUND;
    }
};

#endif
