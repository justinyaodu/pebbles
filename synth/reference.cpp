// Reference implementation of a CPU-based synthesizer.

#include <bitset>
#include <cassert>
#include <vector>

#include "expr.hpp"
#include "spec.hpp"

class Bank {
private:
    // The i'th bit is on iff the bank contains a term whose bitvector
    // of evaluation results is equal to i.
    std::vector<bool> seen;

    // The i'th element stores the evaluation results for the i'th term, where
    // the j'th bit from the right is the evaluation result on example j.
    std::vector<uint32_t> term_results;

    // The i'th element is the left child of the i'th term, or the variable
    // number if the term is a variable.
    std::vector<uint32_t> term_lefts;

    // The (term_lefts.size() - term_rights.size() + i)'th element is the
    // right child of the i'th term. This saves memory because we don't have
    // to store the right child of variable and NOT terms.
    std::vector<uint32_t> term_rights;

public:
    // End indices of each section of terms (variables, NOTs, ANDs, ORs, XORs).
    std::vector<size_t> section_boundaries;

    Bank(size_t max_distinct_terms) :
        seen(std::vector<bool>(max_distinct_terms)) {}

    size_t size() {
        return term_lefts.size();
    }

    uint32_t get_results(size_t i) {
        return term_results[i];
    }

    uint32_t get_left(size_t i) {
        return term_lefts[i];
    }

    uint32_t get_right(size_t i) {
        return term_rights[i + term_rights.size() - term_lefts.size()];
    }

    // Add a variable or NOT term to the bank, unless another term evaluating
    // to the same value has been seen before.
    bool insert_unary(uint32_t result, uint32_t left) {
        assert(result < seen.size());
        if (seen[result]) {
            return false;
        }
        seen[result] = true;
        term_results.push_back(result);
        term_lefts.push_back(left);
        return true;
    }

    // Add an AND, OR, or XOR term to the bank, unless another term evaluating
    // to the same value has been seen before.
    bool insert_binary(uint32_t result, uint32_t left, uint32_t right) {
        assert(result < seen.size());
        if (seen[result]) {
            return false;
        }
        seen[result] = true;
        term_results.push_back(result);
        term_lefts.push_back(left);
        term_rights.push_back(right);
        return true;
    }

    // Indicate the end of a section of terms.
    void end_section() {
        section_boundaries.push_back(term_lefts.size());
    }
};

class Synthesizer {
private:
    Spec spec;

    // The maximum number of observationally distinct terms by depth.
    const size_t max_distinct_terms;

    // Bitmask indicating which bits contain valid examples.
    const uint32_t result_mask;

    // Banks of terms for each depth.
    std::vector<Bank> banks;

public:
    Synthesizer(Spec spec) :
        spec(spec),
        max_distinct_terms(1L << spec.num_examples),
        result_mask(max_distinct_terms - 1) {}

    Expr* reconstruct(uint32_t depth, uint32_t index) {
        assert(banks[depth].section_boundaries.size() == 5);

        uint32_t left = banks[depth].get_left(index);

        if (index < banks[depth].section_boundaries[0]) {
            return Expr::Var(left);
        }

        Expr* left_expr = reconstruct(depth - 1, left);

        if (index < banks[depth].section_boundaries[1]) {
            return Expr::Not(left_expr);
        }

        uint32_t right = banks[depth].get_right(index);
        Expr* right_expr = reconstruct(depth - 1, right);

        if (index < banks[depth].section_boundaries[2]) {
            return Expr::And(left_expr, right_expr);
        }

        if (index < banks[depth].section_boundaries[3]) {
            return Expr::Or(left_expr, right_expr);
        }

        if (index < banks[depth].section_boundaries[4]) {
            return Expr::Xor(left_expr, right_expr);
        }

        // Index out of range.
        assert(false);
        return nullptr;
    }

    Expr* synthesize() {
        for (uint32_t depth = 0; depth <= spec.sol_depth; depth++) {
            std::cerr << "synthesizing depth " << depth << std::endl;

            banks.push_back(Bank(max_distinct_terms));

            // Insert variables with the current depth.
            for (uint32_t i = 0; i < spec.num_vars; i++) {
                if (spec.var_depths[i] == depth) {
                    banks[depth].insert_unary(spec.var_values[i], i);
                }
            }
            banks[depth].end_section();

            if (depth == 0) {
                // Skip NOT, AND, OR, and XOR.
                for (uint32_t i = 0; i < 4; i++) {
                    banks[depth].end_section();
                }
                continue;
            }

            // Did we find a term matching the desired output?
            bool found = false;

            // Synthesize NOT terms.
            for (uint32_t left = 0; !found && left < banks[depth - 1].size(); left++) {
                uint32_t result = ~banks[depth - 1].get_results(left);
                banks[depth].insert_unary(result & result_mask, left);
                if (result == spec.sol_result) {
                    found = true;
                }
            }
            banks[depth].end_section();

            // Synthesize AND terms.
            for (uint32_t left = 0; !found && left < banks[depth - 1].size(); left++) {
                for (uint32_t right = 0; !found && right <= left; right++) {
                    uint32_t result = banks[depth - 1].get_results(left)
                        & banks[depth - 1].get_results(right);
                    banks[depth].insert_binary(result & result_mask, left, right);
                    if (result == spec.sol_result) {
                        found = true;
                    }
                }
            }
            banks[depth].end_section();

            // Synthesize OR terms.
            for (uint32_t left = 0; !found && left < banks[depth - 1].size(); left++) {
                for (uint32_t right = 0; !found && right <= left; right++) {
                    uint32_t result = banks[depth - 1].get_results(left)
                        | banks[depth - 1].get_results(right);
                    banks[depth].insert_binary(result & result_mask, left, right);
                    if (result == spec.sol_result) {
                        found = true;
                    }
                }
            }
            banks[depth].end_section();

            // Synthesize XOR terms.
            for (uint32_t left = 0; !found && left < banks[depth - 1].size(); left++) {
                for (uint32_t right = 0; !found && right <= left; right++) {
                    uint32_t result = banks[depth - 1].get_results(left)
                        ^ banks[depth - 1].get_results(right);
                    banks[depth].insert_binary(result & result_mask, left, right);
                    if (result == spec.sol_result) {
                        found = true;
                    }
                }
            }
            banks[depth].end_section();

            if (found) {
                Expr* expr = reconstruct(depth, banks[depth].size() - 1);

                // This expression might not be the desired depth, but we can
                // AND it with itself and NOT it twice until it's deep enough.
                uint32_t remaining_depth = spec.sol_depth - depth;
                if (remaining_depth % 2 == 1) {
                    expr = Expr::And(expr, expr);
                    remaining_depth--;
                }
                while (remaining_depth > 0) {
                    expr = Expr::Not(Expr::Not(expr));
                    remaining_depth -= 2;
                }

                return expr;
            }
        }

        return nullptr;
    }

    void print_banks(std::ostream &out) {
        for (size_t depth = 0; depth < banks.size(); depth++) {
            out << "depth " << depth << ":\n";
            for (size_t i = 0; i < banks[depth].size(); i++) {
                out << "\t";
                reconstruct(depth, i)->print(out, &spec.var_names);
                out << "\n";
            }
        }
    }
};

int main(void) {
    Spec spec(
        2,
        4,
        std::vector<std::string> { "b1", "b2" },
        std::vector<uint32_t> { 0, 0 },
        std::vector<uint32_t> { 0b0011, 0b0101 },
        0b1011,
        2
    );
    std::cout << spec << std::endl;

    Synthesizer synthesizer(spec);

    Expr* expr = synthesizer.synthesize();
    if (expr == nullptr) {
        std::cout << "no solution" << std::endl;
    } else {
        std::cout << "solution: ";
        expr->print(std::cout, &spec.var_names);
        std::cout << std::endl;
    }

    synthesizer.print_banks(std::cout);
}
