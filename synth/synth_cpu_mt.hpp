// Multi-threaded CPU-based synthesizer.

#ifndef SYNTH_CPU_MT_H
#define SYNTH_CPU_MT_H

#include <cstdint>
#include <cstring>
#include <iostream>
#include <omp.h>

#include "alloc.hpp"
#include "bitset.hpp"
#include "expr.hpp"
#include "spec.hpp"
#include "synth_cpu.hpp"
#include "timer.hpp"

class Synthesizer : AbstractSynthesizer {
private:
    // The i'th bit is on iff the bank contains a term whose bitvector
    // of evaluation results is equal to i.
    ThreadSafeBitset seen;

public:
    Synthesizer(Spec spec) : AbstractSynthesizer(spec),
            seen(ThreadSafeBitset(max_distinct_terms)) {
        // Cancellation must be enabled for best performance, to enable early
        // termination of parallel loops when a solution is found.
        assert(omp_get_cancellation());
        std::cerr << "Using multi-threaded CPU synthesizer" << std::endl;
    }

    // Return an Expr satisfying spec, or nullptr if it cannot be found.
    const Expr* synthesize() {
        int64_t sol_index = NOT_FOUND;

        for (int32_t height = 0; height <= spec.sol_height; height++) {

// Do the specified pass, and break out of the loop if a solution was found.
#define DO_PASS(TYPE)                       \
{                                           \
    int64_t prev_num_terms = num_terms;     \
    std::cerr << "height " << height        \
        << ", " #TYPE " pass" << std::endl;    \
                                            \
    Timer timer;                            \
    sol_index = pass_ ## TYPE(height);      \
    uint64_t ms = timer.ms();               \
    record_pass(PassType::TYPE, height);    \
                                            \
    std::cerr << ms << " ms, "              \
        << (num_terms - prev_num_terms) << " new term(s), "   \
        << num_terms << " total term(s)"    \
        << std::endl;                       \
                                            \
    if (sol_index != NOT_FOUND) {           \
        break;                              \
    }                                       \
}

            DO_PASS(Variable);

            if (height == 0) {
                continue;
            }

            DO_PASS(Not);
            DO_PASS(And);
            DO_PASS(Or);
            DO_PASS(Xor);

#undef DO_PASS
        }

        return sol_index == NOT_FOUND ? nullptr : reconstruct(sol_index);
    }

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

    // Add NOT terms to the bank.
    int64_t pass_Not(int32_t height __attribute__((unused))) {
        // Start at the first term after the preceding NOT pass (if any).
        int64_t lefts_start = 0;
        for (size_t i = 0; i < pass_types.size(); i++) {
            if (pass_types[i] == PassType::Not) {
                lefts_start = pass_ends[i];
            }
        }

        int64_t lefts_end = num_terms;

        int64_t solution = NOT_FOUND;

        #pragma omp parallel
        #pragma omp for
        for (int64_t left = lefts_start; left < lefts_end; left++) {
            uint32_t result = result_mask & (~term_results[left]);
            if (seen.test_and_set(result)) {
                continue;
            }

            uint32_t left_cast = left;
            int64_t index = add_unary_terms(1, &result, &left_cast);

            if (result == spec.sol_result) {
                #pragma omp critical
                { solution = index; }
                #pragma omp cancel for
            }
        }

        return solution;
    }

// See trapezoid_indexing.py for details.
#define TRAPEZOID_LOOP \
for (int64_t b = 0; b < k * (n - k) + (n - k) * (n - k + 1) / 2; b++)

#define TRAPEZOID_LOOP_SETUP \
int64_t left = n - 1 - b % (n - k);     \
int64_t right = b / (n - k);            \
if (right > left) {                     \
    left = n - (left - k) - 1;          \
    right = n - (right - (k + 1)) - 1;  \
}

    // Add AND terms to the bank.
    int64_t pass_And(int32_t height) {
        int64_t k = terms_with_height_start(height - 1);
        int64_t n = terms_with_height_end(height - 1);
        int64_t solution = NOT_FOUND;

        #pragma omp parallel
        #pragma omp for
        TRAPEZOID_LOOP {
            TRAPEZOID_LOOP_SETUP

            uint32_t result = result_mask &
                    (term_results[left] & term_results[right]);
            if (seen.test_and_set(result)) {
                continue;
            }

            uint32_t left_cast = left;
            uint32_t right_cast = right;
            int64_t index = add_binary_terms(1, &result, &left_cast, &right_cast);

            if (result == spec.sol_result) {
                #pragma omp critical
                { solution = index; }
                #pragma omp cancel for
            }
        }

        return solution;
    }

    // Add OR terms to the bank.
    int64_t pass_Or(int32_t height) {
        int64_t k = terms_with_height_start(height - 1);
        int64_t n = terms_with_height_end(height - 1);
        int64_t solution = NOT_FOUND;

        #pragma omp parallel
        #pragma omp for
        TRAPEZOID_LOOP {
            TRAPEZOID_LOOP_SETUP

            uint32_t result = result_mask &
                    (term_results[left] | term_results[right]);
            if (seen.test_and_set(result)) {
                continue;
            }

            uint32_t left_cast = left;
            uint32_t right_cast = right;
            int64_t index = add_binary_terms(1, &result, &left_cast, &right_cast);

            if (result == spec.sol_result) {
                #pragma omp critical
                { solution = index; }
                #pragma omp cancel for
            }
        }

        return solution;
    }

    // Add XOR terms to the bank.
    int64_t pass_Xor(int32_t height) {
        int64_t k = terms_with_height_start(height - 1);
        int64_t n = terms_with_height_end(height - 1);
        int64_t solution = NOT_FOUND;

        #pragma omp parallel
        #pragma omp for
        TRAPEZOID_LOOP {
            TRAPEZOID_LOOP_SETUP

            uint32_t result = result_mask &
                    (term_results[left] ^ term_results[right]);
            if (seen.test_and_set(result)) {
                continue;
            }

            uint32_t left_cast = left;
            uint32_t right_cast = right;
            int64_t index = add_binary_terms(1, &result, &left_cast, &right_cast);

            if (result == spec.sol_result) {
                #pragma omp critical
                { solution = index; }
                #pragma omp cancel for
            }
        }

        return solution;
    }
};

#endif
