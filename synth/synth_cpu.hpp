// Shared code for CPU-based synthesizers.

#ifndef SYNTH_CPU_H
#define SYNTH_CPU_H

#include <cassert>
#include <cstdint>
#include <vector>

#include "alloc.hpp"
#include "expr.hpp"
#include "spec.hpp"

enum class PassType {
    Variable,
    Not,
    And,
    Or,
    Xor
};

class AbstractSynthesizer {
protected:
    // Returned by pass methods when a solution was not found.
    static const int64_t NOT_FOUND = -1;

    Spec spec;

    // The maximum number of observationally distinct terms.
    const size_t max_distinct_terms;

    // Bitmask indicating which bits contain valid examples.
    const uint32_t result_mask;

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

    AbstractSynthesizer(Spec spec) :
        spec(spec),
        max_distinct_terms(1ULL << spec.num_examples),
        result_mask(max_distinct_terms - 1),
        num_terms(0),
        term_results((uint32_t*) alloc(max_distinct_terms * sizeof(uint32_t))),
        term_lefts((uint32_t*) alloc(max_distinct_terms * sizeof(uint32_t))),
        term_rights((uint32_t*) alloc(max_distinct_terms * sizeof(uint32_t))) {}

    ~AbstractSynthesizer() {
        dealloc(term_results, max_distinct_terms * sizeof(uint32_t));
        dealloc(term_lefts, max_distinct_terms * sizeof(uint32_t));
        dealloc(term_rights, max_distinct_terms * sizeof(uint32_t));
    }

    // Called every time a pass is completed.
    void record_pass(PassType type, int32_t height) {
        pass_starts.push_back(pass_ends.size() ? pass_ends.back() : 0);
        pass_ends.push_back(num_terms);
        pass_heights.push_back(height);
        pass_types.push_back(type);
    }

    // Reconstruct the term at the given index in the bank.
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
};

#endif
