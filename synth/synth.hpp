// Shared code for all synthesizers.

#ifndef SYNTH_H
#define SYNTH_H

#include <cassert>
#include <cstdint>
#include <vector>

#include "alloc.hpp"
#include "expr.hpp"
#include "spec.hpp"
#include "timer.hpp"

// The synthesis procedure is organized as a series of passes, where each pass
// indicates what type of terms are being synthesized, and the height of those
// terms.
enum class PassType {
    Variable,
    Not,
    And,
    Or,
    XorCheck,
    XorSynth
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
            case PassType::XorCheck:
            case PassType::XorSynth:
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

    // Return the index of the term with the given result.
    uint32_t find_term_with_result(uint32_t result) {
        // In a multithreaded environment, num_terms might get updated during
        // the execution of the following loop. However, if a term with the
        // specified result is present in the bank when this method is called,
        // we don't need to consider terms that are added concurrently.
        int64_t end = num_terms;
        for (int64_t i = 0; i < end; i++) {
            if (term_results[i] == result) {
                return i;
            }
        }

        // No term with this result exists.
        assert(false);
        return 0;
    }

    virtual int64_t pass_Variable(int32_t height) = 0;
    virtual int64_t pass_Not(int32_t height) = 0;
    virtual int64_t pass_And(int32_t height) = 0;
    virtual int64_t pass_Or(int32_t height) = 0;
    virtual int64_t pass_XorCheck(int32_t height) = 0;
    virtual int64_t pass_XorSynth(int32_t height) = 0;

public:
    // Return an Expr satisfying spec, or nullptr if it cannot be found.
    const Expr* synthesize() {
        int64_t sol_index = NOT_FOUND;
        Timer timer;

        for (int32_t height = 0; height <= spec.sol_height; height++) {

// Do the specified pass, and break out of the loop if a solution was found.
#define DO_PASS(TYPE)                       \
{                                           \
    int64_t prev_num_terms = num_terms;     \
    std::cerr << "height " << height        \
        << ", " #TYPE " pass" << std::endl; \
                                            \
    Timer pass_timer;                       \
    sol_index = pass_ ## TYPE(height);      \
    uint64_t ms = pass_timer.ms();          \
    record_pass(PassType::TYPE, height);    \
                                            \
    std::cerr << "\t" << ms << " ms, "      \
        << (num_terms - prev_num_terms) << " new term(s), " \
        << num_terms << " total term(s)"    \
        << std::endl;                       \
                                            \
    if (sol_index != NOT_FOUND) {           \
        break;                              \
    }                                       \
}

            DO_PASS(Variable);

            // The subsequent passes look for terms in the bank whose height is
            // one less. When the height is 0, we should skip them.
            if (height == 0) {
                continue;
            }

            DO_PASS(Not);
            DO_PASS(XorCheck);

            DO_PASS(And);
            DO_PASS(Or);

            // Only synthesize new terms if we need them for the next iteration.
            if (height < spec.sol_height) {
                DO_PASS(XorSynth);
            }

#undef DO_PASS
        }

        uint64_t ms = timer.ms();
        std::cerr << ms << " ms, "
            << num_terms << " terms" << std::endl;

        return sol_index == NOT_FOUND ? nullptr : reconstruct(sol_index);
    }
};

#endif
