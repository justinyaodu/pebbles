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
    // This is used to avoid inserting new terms that are observationally
    // equivalent to previously inserted terms.
    SingleThreadedBitset seen;

public:
    Synthesizer(Spec spec) : AbstractSynthesizer(spec),
        seen(SingleThreadedBitset(max_distinct_terms)) {}

private:
    // Return the next free index to be used for a new term.
    int64_t alloc_term() {
        return num_terms++;
    }

    // Add a NOT term or variable term to the bank.
    void add_unary_term(uint32_t result, uint32_t left) {
        int64_t index = alloc_term();
        term_results[index] = result;
        term_lefts[index] = left;
    }

    // Add a binary operator term to the bank.
    void add_binary_term(uint32_t result, uint32_t left, uint32_t right) {
        int64_t index = alloc_term();
        term_results[index] = result;
        term_lefts[index] = left;
        term_rights[index] = right;
    }

    // Add variables of the specified height to the bank.
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

    // Synthesize NOT terms.
    int64_t pass_Not(int32_t height) {
        // The operand must be a term whose height is one less than the current
        // height.
        int64_t lefts_start = terms_with_height_start(height - 1);
        int64_t lefts_end = terms_with_height_end(height - 1);

        for (int64_t left = lefts_start; left < lefts_end; left++) {
            uint32_t left_result = term_results[left];
            uint32_t result = result_mask & ~left_result;
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

    // Synthesize binary operator terms (AND, OR, and XOR).
    //
    // We make this a template to avoid duplicating code between the three
    // binary operators, since everything is the same except for the operation
    // being performed.
    //
    // Ideally, this would be an instance method which takes a lambda as a
    // parameter. However, the compiler can't inline the lambda in that case,
    // which makes the code roughly twice as slow.
    template <typename Op>
    friend int64_t pass_binary(Synthesizer &self, int32_t height, Op op) {
        // The right operand must be a term whose height is one less than the
        // current height.
        int64_t rights_start = self.terms_with_height_start(height - 1);
        int64_t rights_end = self.terms_with_height_end(height - 1);

        for (int64_t right = rights_start; right < rights_end; right++) {
            uint32_t right_result = self.term_results[right];

            // The left operand can be any term whose height is less than the
            // current height. Since each binary operator is commutative, we
            // only consider (left, right) pairs where left <= right, to avoid
            // constructing redundant terms.
            for (int64_t left = 0; left <= right; left++) {
                uint32_t left_result = self.term_results[left];
                uint32_t result = self.result_mask & op(left_result, right_result);
                if (self.seen.test_and_set(result)) {
                    continue;
                }

                self.add_binary_term(result, left, right);

                if (result == self.spec.sol_result) {
                    return self.num_terms - 1;
                }
            }
        }

        return Synthesizer::NOT_FOUND;
    }

    int64_t pass_And(int32_t height) {
        auto op = [](uint32_t a, uint32_t b) { return a & b; };
        return pass_binary(*this, height, op);
    }

    int64_t pass_Or(int32_t height) {
        auto op = [](uint32_t a, uint32_t b) { return a | b; };
        return pass_binary(*this, height, op);
    }

    int64_t pass_Xor(int32_t height) {
        auto op = [](uint32_t a, uint32_t b) { return a ^ b; };
        return pass_binary(*this, height, op);
    }
};

#endif
