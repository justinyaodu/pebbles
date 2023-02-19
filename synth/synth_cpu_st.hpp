// Single-threaded CPU-based synthesizer.

#ifndef SYNTH_CPU_ST_H
#define SYNTH_CPU_ST_H

#include <cstdint>

#include "bitset.hpp"
#include "expr.hpp"
#include "spec.hpp"
#include "synth.hpp"
#include "timer.hpp"

class Synthesizer : public AbstractSynthesizer {
private:
    // The i'th bit is on iff the bank contains a term whose bitvector
    // of evaluation results is equal to i.
    SingleThreadedBitset seen;

public:
    Synthesizer(Spec spec) : AbstractSynthesizer(spec),
        seen(SingleThreadedBitset(max_distinct_terms)) {}

private:
    int64_t alloc_term() {
        return num_terms++;
    }

    void add_unary_term(uint32_t result, uint32_t left) {
        int64_t index = alloc_term();
        term_results[index] = result;
        term_lefts[index] = left;
    }

    void add_binary_term(uint32_t result, uint32_t left, uint32_t right) {
        int64_t index = alloc_term();
        term_results[index] = result;
        term_lefts[index] = left;
        term_rights[index] = right;
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

            add_unary_term(result, i);

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

        int64_t lefts_end = terms_with_height_end(height - 1);

        for (int64_t left = lefts_start; left < lefts_end; left++) {
            uint32_t result = result_mask & (~term_results[left]);
            if (seen.test_and_set(result)) {
                continue;
            }

            add_unary_term(result, left);

            if (result == spec.sol_result) {
                return num_terms - 1;
            }
        }

        return NOT_FOUND;
    }

    // Add AND terms to the bank.
    int64_t pass_And(int32_t height) {
        int64_t lefts_start = terms_with_height_start(height - 1);
        int64_t lefts_end = terms_with_height_end(height - 1);

        for (int64_t left = lefts_start; left < lefts_end; left++) {
            // AND uses < instead of <= because ANDing a term with itself is useless.
            for (int64_t right = 0; right < left; right++) {
                uint32_t result = result_mask &
                        (term_results[left] & term_results[right]);
                if (seen.test_and_set(result)) {
                    continue;
                }

                add_binary_term(result, left, right);

                if (result == spec.sol_result) {
                    return num_terms - 1;
                }
            }
        }

        return NOT_FOUND;
    }

    // Add OR terms to the bank.
    int64_t pass_Or(int32_t height) {
        int64_t lefts_start = terms_with_height_start(height - 1);
        int64_t lefts_end = terms_with_height_end(height - 1);

        for (int64_t left = lefts_start; left < lefts_end; left++) {
            // OR uses < instead of <= because ORing a term with itself is useless.
            for (int64_t right = 0; right < left; right++) {
                uint32_t result = result_mask &
                        (term_results[left] | term_results[right]);
                if (seen.test_and_set(result)) {
                    continue;
                }

                add_binary_term(result, left, right);

                if (result == spec.sol_result) {
                    return num_terms - 1;
                }
            }
        }

        return NOT_FOUND;
    }

    // Add XOR terms to the bank.
    int64_t pass_Xor(int32_t height) {
        int64_t lefts_start = terms_with_height_start(height - 1);
        int64_t lefts_end = terms_with_height_end(height - 1);

        for (int64_t left = lefts_start; left < lefts_end; left++) {
            // XOR uses <= instead of < because XORing a term with itself gives 0.
            for (int64_t right = 0; right <= left; right++) {
                uint32_t result = result_mask &
                        (term_results[left] ^ term_results[right]);
                if (seen.test_and_set(result)) {
                    continue;
                }

                add_binary_term(result, left, right);

                if (result == spec.sol_result) {
                    return num_terms - 1;
                }
            }
        }

        return NOT_FOUND;
    }
};

#endif
