// Reference implementation of a CPU-based synthesizer.

#include <bitset>
#include <vector>

#include "expr.hpp"
#include "spec.hpp"

class Term {
public:
    // The i'th bit is the result of evaluating this term on example i.
    uint32_t result;

    // The index of this term's left child in the preceding bank.
    uint32_t left;

    // The index of this term's right child in the preceding bank.
    uint32_t right;

    Term(uint32_t result, uint32_t left, uint32_t right) :
        result(result), left(left), right(right) {}
};

class Bank {
private:
    // The i'th bit is on iff the bank contains a term whose bitvector
    // of evaluation results is equal to i.
    std::bitset<1L << 32> bitset;

    std::vector<Term> terms;

    // End indices of each section of terms (variables, NOTs, ANDs, ORs, XORs).
    std::vector<size_t> section_boundaries;

public:
    // Add a term to the bank, unless another term evaluating to the same
    // value has been seen before.
    bool insert(Term &term) {
        if (this->bitset[term.result]) {
            return false;
        }
        this->bitset[term.result] = true;
        this->terms.push_back(term);
        return true;
    }

    // Indicate the end of a section of terms.
    void end_section() {
        this->section_boundaries.push_back(this->terms.size());
    }
};

class Synthesizer {
    Spec &spec;

    // Banks of terms for each depth.
    std::vector<Bank> banks;

    Synthesizer(Spec spec) : spec(spec) {}
};

int main(void) {
    // No synthesizer yet - testing boolean expressions.

    Expr x0 = Expr::Var(0);
    Expr x1 = Expr::Var(1);
    Expr x2 = Expr::Var(2);
    Expr x3 = Expr::Var(3);
    Expr not_zero = Expr::Not(&x0);
    Expr not_zero_xor_one = Expr::Xor(&not_zero, &x1);
    Expr two_and_three = Expr::And(&x2, &x3);
    Expr top = Expr::Or(&not_zero_xor_one, &two_and_three);
    std::cout << top << std::endl;

    // x0 = false, x1 = false, x2 = true, x3 = false
    std::vector<bool> var_values { false, false, true, false };
    std::cout << top.eval(var_values) << std::endl;
}
