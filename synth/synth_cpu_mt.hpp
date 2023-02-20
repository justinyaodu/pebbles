// Multi-threaded CPU-based synthesizer.

#ifndef SYNTH_CPU_MT_H
#define SYNTH_CPU_MT_H

#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <omp.h>

#include "bitset.hpp"
#include "expr.hpp"
#include "spec.hpp"
#include "synth.hpp"

// Set experimentally.
#define TILE_SIZE 64

class Synthesizer : public AbstractSynthesizer {
private:
    // The i'th bit is on iff the bank contains a term whose bitvector
    // of evaluation results is equal to i.
    ThreadSafeBitset seen;

public:
    Synthesizer(Spec spec) : AbstractSynthesizer(spec),
            seen(ThreadSafeBitset(max_distinct_terms)) {}

private:
    int64_t alloc_terms(int64_t count) {
        return __atomic_fetch_add(&num_terms, count, __ATOMIC_SEQ_CST);
    }

    int64_t add_unary_terms(int64_t count, uint32_t *results, uint32_t *lefts) {
        int64_t start = alloc_terms(count);
        memcpy(&term_results[start], results, count * sizeof(uint32_t));
        memcpy(&term_lefts[start], lefts, count * sizeof(uint32_t));
        return start;
    }

    int64_t add_binary_terms(int64_t count, uint32_t *results, uint32_t *lefts,
            uint32_t *rights) {
        int64_t start = alloc_terms(count);
        memcpy(&term_results[start], results, count * sizeof(uint32_t));
        memcpy(&term_lefts[start], lefts, count * sizeof(uint32_t));
        memcpy(&term_rights[start], rights, count * sizeof(uint32_t));
        return start;
    }

    // Add variables to the bank.
    int64_t pass_Variable(int32_t height) {
        for (size_t i = 0; i < spec.num_vars; i++) {
            if (spec.var_heights[i] != height) {
                continue;
            }

            uint32_t result = result_mask & spec.var_values[i];
            if (seen.test_and_set(result)) {
                continue;
            }

            uint32_t left_cast = i;
            add_unary_terms(1, &result, &left_cast);

            if (result == spec.sol_result) {
                return num_terms - 1;
            }
        }

        return NOT_FOUND;
    }

    // Synthesize NOT terms.
    int64_t pass_Not(int32_t height) {
        int64_t all_lefts_start = terms_with_height_start(height - 1);
        int64_t all_lefts_end = terms_with_height_end(height - 1);

        int64_t solution = NOT_FOUND;

        #pragma omp parallel for
        for (int64_t lefts_tile = all_lefts_start / TILE_SIZE;
                lefts_tile < CEIL_DIV(all_lefts_end, TILE_SIZE);
                lefts_tile++) {
            if (solution != NOT_FOUND) {
                continue;
            }

            int32_t batch_size = 0;
            uint32_t batch_results[TILE_SIZE];
            uint32_t batch_lefts[TILE_SIZE];

            for (int64_t left = std::max(lefts_tile * TILE_SIZE, all_lefts_start);
                    left < std::min((lefts_tile + 1) * TILE_SIZE, all_lefts_end);
                    left++) {
                uint32_t result = result_mask & (~term_results[left]);
                if (seen.test_and_set(result)) {
                    continue;
                }

                batch_results[batch_size] = result;
                batch_lefts[batch_size] = left;
                batch_size++;
            }

            if (batch_size == 0) {
                continue;
            }

            int64_t bank_index = add_unary_terms(batch_size, batch_results, batch_lefts);
            for (int32_t i = 0; i < batch_size; i++) {
                if (batch_results[i] == spec.sol_result) {
                    // No synchronization needed, because if two threads find a
                    // solution simultaneously, it doesn't matter which we use.
                    solution = bank_index + i;
                }
            }
        }

        return solution;
    }

    // Synthesize AND terms.
    // The code is exactly the same for AND, OR, and XOR, except for the bitwise
    // operation used. I tried specifying the operation with a lambda/
    // function pointer/etc., but it hurt performance because I couldn't coerce
    // the compiler into inlining it.
    int64_t pass_And(int32_t height) {
        int64_t all_lefts_end = terms_with_height_end(height - 1);

        int64_t all_rights_start = terms_with_height_start(height - 1);
        int64_t all_rights_end = all_lefts_end;

        int64_t solution = NOT_FOUND;

        // See trapezoid_indexing.py for details.
        int64_t k = all_rights_start / TILE_SIZE;
        int64_t n = CEIL_DIV(all_rights_end, TILE_SIZE);
        #pragma omp parallel for
        for (int64_t b = 0; b < k * (n - k) + (n - k) * (n - k + 1) / 2; b++) {
            if (solution != NOT_FOUND) {
                continue;
            }

            int64_t lefts_tile = b / (n - k);
            int64_t rights_tile = n - 1 - b % (n - k);
            if (lefts_tile > rights_tile) {
                lefts_tile = n - (lefts_tile - (k + 1)) - 1;
                rights_tile = n - (rights_tile - k) - 1;
            }

            int32_t batch_size = 0;
            uint32_t batch_results[TILE_SIZE * TILE_SIZE];
            uint32_t batch_lefts[TILE_SIZE * TILE_SIZE];
            uint32_t batch_rights[TILE_SIZE * TILE_SIZE];

            // No max needed on next line because all_lefts_start = 0.
            for (int64_t left = lefts_tile * TILE_SIZE;
                    left < std::min((lefts_tile + 1) * TILE_SIZE, all_lefts_end);
                    left++) {
                uint32_t left_result = term_results[left];
                for (int64_t right = std::max(rights_tile * TILE_SIZE, all_rights_start);
                        right < std::min((rights_tile + 1) * TILE_SIZE, all_rights_end);
                        right++) {
                    uint32_t right_result = term_results[right];
                    uint32_t result = result_mask & (left_result & right_result);
                    if (seen.test_and_set(result)) {
                        continue;
                    }

                    batch_results[batch_size] = result;
                    batch_lefts[batch_size] = left;
                    batch_rights[batch_size] = right;
                    batch_size++;
                }
            }

            if (batch_size == 0) {
                continue;
            }

            int64_t bank_index = add_binary_terms(
                    batch_size, batch_results, batch_lefts, batch_rights);
            for (int32_t i = 0; i < batch_size; i++) {
                if (batch_results[i] == spec.sol_result) {
                    // No synchronization needed, because if two threads find a
                    // solution simultaneously, it doesn't matter which we use.
                    solution = bank_index + i;
                }
            }
        }

        return solution;
    }

    int64_t pass_Or(int32_t height) {
        int64_t all_lefts_end = terms_with_height_end(height - 1);

        int64_t all_rights_start = terms_with_height_start(height - 1);
        int64_t all_rights_end = all_lefts_end;

        int64_t solution = NOT_FOUND;

        // See trapezoid_indexing.py for details.
        int64_t k = all_rights_start / TILE_SIZE;
        int64_t n = CEIL_DIV(all_rights_end, TILE_SIZE);
        #pragma omp parallel for
        for (int64_t b = 0; b < k * (n - k) + (n - k) * (n - k + 1) / 2; b++) {
            if (solution != NOT_FOUND) {
                continue;
            }

            int64_t lefts_tile = b / (n - k);
            int64_t rights_tile = n - 1 - b % (n - k);
            if (lefts_tile > rights_tile) {
                lefts_tile = n - (lefts_tile - (k + 1)) - 1;
                rights_tile = n - (rights_tile - k) - 1;
            }

            int32_t batch_size = 0;
            uint32_t batch_results[TILE_SIZE * TILE_SIZE];
            uint32_t batch_lefts[TILE_SIZE * TILE_SIZE];
            uint32_t batch_rights[TILE_SIZE * TILE_SIZE];

            // No max needed on next line because all_lefts_start = 0.
            for (int64_t left = lefts_tile * TILE_SIZE;
                    left < std::min((lefts_tile + 1) * TILE_SIZE, all_lefts_end);
                    left++) {
                uint32_t left_result = term_results[left];
                for (int64_t right = std::max(rights_tile * TILE_SIZE, all_rights_start);
                        right < std::min((rights_tile + 1) * TILE_SIZE, all_rights_end);
                        right++) {
                    uint32_t right_result = term_results[right];
                    uint32_t result = result_mask & (left_result | right_result);
                    if (seen.test_and_set(result)) {
                        continue;
                    }

                    batch_results[batch_size] = result;
                    batch_lefts[batch_size] = left;
                    batch_rights[batch_size] = right;
                    batch_size++;
                }
            }

            if (batch_size == 0) {
                continue;
            }

            int64_t bank_index = add_binary_terms(
                    batch_size, batch_results, batch_lefts, batch_rights);
            for (int32_t i = 0; i < batch_size; i++) {
                if (batch_results[i] == spec.sol_result) {
                    // No synchronization needed, because if two threads find a
                    // solution simultaneously, it doesn't matter which we use.
                    solution = bank_index + i;
                }
            }
        }

        return solution;
    }

    int64_t pass_Xor(int32_t height) {
        int64_t all_lefts_end = terms_with_height_end(height - 1);

        int64_t all_rights_start = terms_with_height_start(height - 1);
        int64_t all_rights_end = all_lefts_end;

        int64_t solution = NOT_FOUND;

        // See trapezoid_indexing.py for details.
        int64_t k = all_rights_start / TILE_SIZE;
        int64_t n = CEIL_DIV(all_rights_end, TILE_SIZE);
        #pragma omp parallel for
        for (int64_t b = 0; b < k * (n - k) + (n - k) * (n - k + 1) / 2; b++) {
            if (solution != NOT_FOUND) {
                continue;
            }

            int64_t lefts_tile = b / (n - k);
            int64_t rights_tile = n - 1 - b % (n - k);
            if (lefts_tile > rights_tile) {
                lefts_tile = n - (lefts_tile - (k + 1)) - 1;
                rights_tile = n - (rights_tile - k) - 1;
            }

            int32_t batch_size = 0;
            uint32_t batch_results[TILE_SIZE * TILE_SIZE];
            uint32_t batch_lefts[TILE_SIZE * TILE_SIZE];
            uint32_t batch_rights[TILE_SIZE * TILE_SIZE];

            // No max needed on next line because all_lefts_start = 0.
            for (int64_t left = lefts_tile * TILE_SIZE;
                    left < std::min((lefts_tile + 1) * TILE_SIZE, all_lefts_end);
                    left++) {
                uint32_t left_result = term_results[left];
                for (int64_t right = std::max(rights_tile * TILE_SIZE, all_rights_start);
                        right < std::min((rights_tile + 1) * TILE_SIZE, all_rights_end);
                        right++) {
                    uint32_t right_result = term_results[right];
                    uint32_t result = result_mask & (left_result ^ right_result);
                    if (seen.test_and_set(result)) {
                        continue;
                    }

                    batch_results[batch_size] = result;
                    batch_lefts[batch_size] = left;
                    batch_rights[batch_size] = right;
                    batch_size++;
                }
            }

            if (batch_size == 0) {
                continue;
            }

            int64_t bank_index = add_binary_terms(
                    batch_size, batch_results, batch_lefts, batch_rights);
            for (int32_t i = 0; i < batch_size; i++) {
                if (batch_results[i] == spec.sol_result) {
                    // No synchronization needed, because if two threads find a
                    // solution simultaneously, it doesn't matter which we use.
                    solution = bank_index + i;
                }
            }
        }

        return solution;
    }
};

#endif
