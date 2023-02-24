#ifndef SYNTH_GPU_H
#define SYNTH_GPU_H

#include <cstdint>
#include <iostream>

#include "bitset_gpu.cu"
#include "expr.hpp"
#include "spec.hpp"
#include "synth.hpp"

#define TILE_SIZE 8
#define BLOCK_SIZE (TILE_SIZE * TILE_SIZE)
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

// We want to check the total number of new terms and exit early if there aren't
// any. After the upsweep, we already have the total number of new terms, so we
// can check that and return early. This saves time because we don't have to run
// the downsweep unless it's actually needed.
//
// See inclusive_scan.py for a standalone implementation.

__device__ inline void inclusive_scan_upsweep_and_syncthreads(
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
}

__device__ inline void inclusive_scan_downsweep_and_syncthreads(
    uint32_t* values,
    uint32_t length
) {
    uint32_t step, idx;
    for (step = length / 4, idx = threadIdx.x * (2 * step) + (3 * step) - 1;
            step > 0;
            step /= 2, idx /= 2, __syncthreads()) {
        if (idx < length) {
            values[idx] += values[idx - step];
        }
    }
}

// TODO: update this to match pass_binary
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

    inclusive_scan_upsweep_and_syncthreads(new_count, BLOCK_SIZE);

    uint32_t batch_size = new_count[BLOCK_SIZE - 1];
    if (batch_size == 0) {
        return;
    }

    inclusive_scan_downsweep_and_syncthreads(new_count, BLOCK_SIZE);

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

// For all threads where is_new is true, write value to batch_buffer[batch_idx].
// Then copy batch_size values from batch_buffer to dest. This ensures that
// the threads in a warp write to the same contiguous regions of global memory.
//
// This provides a ~2x speedup over the simpler approach of having each thread
// write its value directly to global memory.
__device__ inline void compacted_write(
    uint32_t* dest,
    uint32_t* batch_buffer,
    uint32_t batch_size,
    uint32_t batch_idx,
    uint32_t is_new,
    uint32_t value
) {
    if (is_new) {
        batch_buffer[batch_idx] = value;
    }
    __syncthreads();

    if (threadIdx.x < batch_size) {
        dest[threadIdx.x] = batch_buffer[threadIdx.x];
    }
    __syncthreads();
}

template <typename Op>
__global__ void pass_binary(
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
    uint32_t all_rights_end,
    Op op
) {
    RETURN_IF_FOUND_SOL;

    uint32_t result;
    uint32_t left;
    uint32_t right;
    uint32_t is_new = 0;

    // Synthesize new terms.
    {
        __shared__ uint32_t lefts_tile_results[TILE_SIZE];
        __shared__ uint32_t rights_tile_results[TILE_SIZE];

        uint32_t lefts_tile = blockIdx.x;
        uint32_t rights_tile = n - 1 - blockIdx.y;
        if (lefts_tile > rights_tile) {
            lefts_tile = n - (lefts_tile - (k + 1)) - 1;
            rights_tile = n - (rights_tile - k) - 1;
        }

        uint32_t lefts_base = lefts_tile * TILE_SIZE;
        uint32_t rights_base = rights_tile * TILE_SIZE;

        // TODO: unclear if tiling actually improves performance here.
        if (threadIdx.x < TILE_SIZE) {
            lefts_tile_results[threadIdx.x] = term_results[lefts_base + threadIdx.x];
            rights_tile_results[threadIdx.x] = term_results[rights_base + threadIdx.x];
        }
        __syncthreads();


        uint32_t lefts_offset = threadIdx.x / TILE_SIZE;
        uint32_t rights_offset = threadIdx.x % TILE_SIZE;
        left = lefts_base + lefts_offset;
        right = rights_base + rights_offset;

        if (left < all_lefts_end && all_rights_start <= right && right < all_rights_end) {
            uint32_t left_result = lefts_tile_results[lefts_offset];
            uint32_t right_result = rights_tile_results[rights_offset];
            result = result_mask & op(left_result, right_result);
            // Experimentally, checking result != left_result && result != right_result
            // before checking the bitset doesn't improve performance.
            if (!GPUBitset_test_and_set(seen, result)) {
                is_new = 1;
            }
        }
        __syncthreads();
    }

    // Find the total number of new terms, and the index of the current term
    // in the batch (assuming the current term is new).
    uint32_t batch_size;
    uint32_t batch_idx;
    {
        __shared__ uint32_t new_count[BLOCK_SIZE];

        new_count[threadIdx.x] = is_new;
        __syncthreads();

        inclusive_scan_upsweep_and_syncthreads(new_count, BLOCK_SIZE);

        batch_size = new_count[BLOCK_SIZE - 1];
        if (batch_size == 0) {
            return;
        }
        __syncthreads();

        inclusive_scan_downsweep_and_syncthreads(new_count, BLOCK_SIZE);

        batch_idx = new_count[threadIdx.x] - 1;
        __syncthreads();
    }

    // Find the offset into the bank where new terms should be inserted.
    uint32_t bank_segment_start;
    {
        __shared__ uint32_t shared_bank_segment_start;

        if (threadIdx.x == 0) {
            shared_bank_segment_start = atomicAdd(&state->num_terms, batch_size);
        }
        __syncthreads();

        bank_segment_start = shared_bank_segment_start;
        __syncthreads();
    }

    // Check whether this term is a valid solution.
    if (result == sol_result) {
        state->found_sol = true;
        state->sol_idx = bank_segment_start + batch_idx;
    }

    {
        __shared__ uint32_t batch_buffer[BLOCK_SIZE];

        compacted_write(
            &term_results[bank_segment_start],
            batch_buffer,
            batch_size,
            batch_idx,
            is_new,
            result
        );

        compacted_write(
            &term_lefts[bank_segment_start],
            batch_buffer,
            batch_size,
            batch_idx,
            is_new,
            left
        );

        compacted_write(
            &term_rights[bank_segment_start],
            batch_buffer,
            batch_size,
            batch_idx,
            is_new,
            right
        );
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

    ~Synthesizer() {
        gpuAssert(cudaFree(device_pass_state));
    }

private:

#define DEVICE_SYNC_AND_RETURN_IF_SOLUTION_FOUND(SELF) \
    do {                                    \
        cudaDeviceSynchronize();            \
        DevicePassState state(0);           \
        gpuAssert(cudaMemcpy(               \
                &state,                     \
                (SELF).device_pass_state,     \
                sizeof(DevicePassState),    \
                cudaMemcpyDeviceToHost));   \
        (SELF).num_terms = state.num_terms;   \
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
        DEVICE_SYNC_AND_RETURN_IF_SOLUTION_FOUND(*this);
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
        DEVICE_SYNC_AND_RETURN_IF_SOLUTION_FOUND(*this);
        return NOT_FOUND;
    }

// These methods aren't actually used outside of this class, but nvcc requires
// these to be public to use lambdas.
public:
    template <typename Op>
    friend int64_t pass_binary(Synthesizer &self, int32_t height, Op op) {
        int64_t all_lefts_end = self.terms_with_height_end(height - 1);

        int64_t all_rights_start = self.terms_with_height_start(height - 1);
        int64_t all_rights_end = all_lefts_end;

        // Here, we also use trapezoidal indexing as described in
        // synth_cpu_mt.hpp. However, we make some modifications to better suit
        // GPU execution.
        //
        // The width of the 2D space of tiles can be up to 2^32 (the maximum
        // number of distinct terms we can store) divided by TILE_SIZE.
        // Thus, a 1D index into this space could be up to (2^32 / TILE_SIZE)
        // squared, which would have to be stored in a 64-bit integer. However,
        // 64-bit integer division is very slow on CUDA cores, since it is
        // emulated with 32-bit integer operations.
        //
        // Instead, we can start with coordinates on the rectangular region, and
        // map those to coordinates on the trapezoid as before. The easiest way
        // to give each thread block a distinct 2D coordinate is to use the
        // block index. The x coordinate of a block is the vertical position,
        // starting at 0 for the topmost row of blocks and counting down. The y
        // coordinate of a block is the horizontal position, starting at 0 for
        // the rightmost column of blocks in the trapezoid and counting leftward.
        //
        // There is also a limit on the maximum y-coordinate of a block, so we
        // might not be able to fit the entire trapezoidal region into a single
        // kernel launch. Instead, we split the trapezoid into multiple narrower
        // vertical strips. Conveniently, these vertical strips are also
        // trapezoids, and can be indexed in the same manner.

        // Here, k is the index of the first column in the current vertical
        // strip, n is the index after the last column in the current vertical
        // strip, and max_n is the index after the last column in the full
        // trapezoidal region.
        int64_t max_n = CEIL_DIV(all_rights_end, TILE_SIZE);
        for (int64_t k = all_rights_start / TILE_SIZE; k < max_n; k += MAX_GRID_DIM_Y) {
            int64_t n = std::min(k + MAX_GRID_DIM_Y, max_n);

            dim3 dim_grid(CEIL_DIV((k + 1) + n, 2), n - k);
            dim3 dim_block(BLOCK_SIZE);

            pass_binary<<<dim_grid, dim_block>>>(
                self.device_pass_state,
                self.result_mask,
                self.seen,
                self.spec.sol_result,
                self.term_results,
                self.term_lefts,
                self.term_rights,
                k,
                n,
                all_lefts_end,
                all_rights_start,
                all_rights_end,
                op
            );
            DEVICE_SYNC_AND_RETURN_IF_SOLUTION_FOUND(self);
        }
        return NOT_FOUND;
    }

    int64_t pass_And(int32_t height) {
        auto op = [] __host__ __device__ (uint32_t a, uint32_t b) { return a & b; };
        return pass_binary(*this, height, op);
    }

    int64_t pass_Or(int32_t height) {
        auto op = [] __host__ __device__ (uint32_t a, uint32_t b) { return a | b; };
        return pass_binary(*this, height, op);
    }

    int64_t pass_Xor(int32_t height) {
        auto op = [] __host__ __device__ (uint32_t a, uint32_t b) { return a ^ b; };
        return pass_binary(*this, height, op);
    }
};

#endif
