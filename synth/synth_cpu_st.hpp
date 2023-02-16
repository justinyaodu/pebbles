// Single-threaded CPU-based synthesizer.

#ifndef SYNTH_CPU_ST_H
#define SYNTH_CPU_ST_H

#include <cassert>
#include <cstdint>
#include <iostream>

#include "alloc.hpp"
#include "bitset.hpp"
#include "expr.hpp"
#include "spec.hpp"
#include "timer.hpp"

enum class PassType {
    Variable,
    Not,
    And,
    Or,
    Xor,
};

class Synthesizer {
private:
    // Returned by pass methods when a solution was not found.
    static const int64_t NOT_FOUND = -1;

    Spec spec;

    // The maximum number of observationally distinct terms.
    const size_t max_distinct_terms;

    // Bitmask indicating which bits contain valid examples.
    const uint32_t result_mask;

    // The i'th bit is on iff the bank contains a term whose bitvector
    // of evaluation results is equal to i.
    SingleThreadedBitset seen;

    // Number of terms in the bank.
    int64_t num_terms;

    // The i'th element stores the evaluation results for the i'th term, where
    // the j'th bit from the right is the evaluation result on example j.
    uint32_t* const term_results;

    // The i'th element is the left child of the i'th term, or the variable
    // number if the term is a variable.
    uint32_t* const term_lefts;

    // The i'th element is the right child of the i'th term, or undefined if
    // the term is a variable or a NOT.
    uint32_t* const term_rights;

    // The i'th element is the size of the bank when the i'th pass started,
    // or equivalently, the index of the first element in the i'th pass.
    std::vector<int64_t> pass_starts;

    // The i'th element is the size of the bank when the i'th pass ended,
    // or equivalently, the index after the last element in the i'th pass.
    std::vector<int64_t> pass_ends;

    // The height of the terms enumerated in the i'th pass.
    std::vector<int32_t> pass_heights;

    // The type of the i'th pass.
    std::vector<PassType> pass_types;

    // Called every time a pass is completed.
    void record_pass(PassType type, int32_t height) {
        pass_starts.push_back(pass_ends.size() ? pass_ends.back() : 0);
        pass_ends.push_back(num_terms);
        pass_heights.push_back(height);
        pass_types.push_back(type);
    }

    const Expr* reconstruct(int64_t index) {
        assert(0 <= index && index < num_terms);

        size_t pass = 0;
        while (true) {
            assert(pass < pass_ends.size());
            if (index < pass_ends[pass]) {
                break;
            }
            pass++;
        }

        switch (pass_types[pass]) {
            case PassType::Variable:
                return Expr::Var(term_lefts[index]);
            case PassType::Not:
                return Expr::Not(reconstruct(term_lefts[index]));
            case PassType::And:
                return Expr::And(
                        reconstruct(term_lefts[index]),
                        reconstruct(term_rights[index]));
            case PassType::Or:
                return Expr::Or(
                        reconstruct(term_lefts[index]),
                        reconstruct(term_rights[index]));
            case PassType::Xor:
                return Expr::Xor(
                        reconstruct(term_lefts[index]),
                        reconstruct(term_rights[index]));
        }

        assert(false);
        return nullptr;
    }

public:
    Synthesizer(Spec spec) :
        spec(spec),
        max_distinct_terms(1ULL << spec.num_examples),
        result_mask(max_distinct_terms - 1),
        seen(SingleThreadedBitset(max_distinct_terms)),
        num_terms(0),
        term_results((uint32_t*) alloc(max_distinct_terms * sizeof(uint32_t))),
        term_lefts((uint32_t*) alloc(max_distinct_terms * sizeof(uint32_t))),
        term_rights((uint32_t*) alloc(max_distinct_terms * sizeof(uint32_t))) {}

    ~Synthesizer() {
        dealloc(term_results, max_distinct_terms * sizeof(uint32_t));
        dealloc(term_lefts, max_distinct_terms * sizeof(uint32_t));
        dealloc(term_rights, max_distinct_terms * sizeof(uint32_t));
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

    // Return the index of the first term with the given height.
    int64_t terms_with_height_start(int32_t height) {
        int64_t index = 0;
        for (size_t i = 0; i < pass_types.size(); i++) {
            if (pass_heights[i] == height) {
                index = pass_starts[i];
                break;
            }
        }
        return index;
    }

    // Return the index after the last term with the given height.
    int64_t terms_with_height_end(int32_t height) {
        int64_t index = 0;
        for (size_t i = 0; i < pass_types.size(); i++) {
            if (pass_heights[i] == height) {
                index = pass_ends[i];
            }
        }
        return index;
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

        int64_t lefts_end = num_terms;

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
