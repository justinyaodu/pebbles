// Reference implementation of a CPU-based synthesizer.

#include <bitset>
#include <cassert>
#include <vector>
#include <string>
#include <iostream>
#include <filesystem>
namespace fs = std::filesystem;

#include "expr.hpp"
#include "spec.hpp"
#include "parser.hpp"

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
        /*if (result == 169) {
            std::cout << "Trying to insert a matching solution (not)!" << std::endl;
            std::cout << "Seen before: " << seen[result] << std::endl;
        }*/
        if (result >= seen.size()) {
            std::cout << result << std::endl;
            std::cout << seen.size() << std::endl;
        }
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
        /*if (result == 169) {
            std::cout << "Trying to insert a matching solution!" << std::endl;
            std::cout << "Seen before: " << seen[result] << std::endl;
        }*/
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

    // The maximum number of observationally distinct terms by height.
    const size_t max_distinct_terms;

    // Bitmask indicating which bits contain valid examples.
    const uint32_t result_mask;

    // Banks of terms for each height.
    std::vector<Bank> banks;

public:
    Synthesizer(Spec spec) :
        spec(spec),
        max_distinct_terms(1ULL << spec.num_examples),
        result_mask(max_distinct_terms - 1) {}

    const Expr* reconstruct(int32_t height, uint32_t index) {
        assert(banks[height].section_boundaries.size() == 5);

        uint32_t left = banks[height].get_left(index);

        if (index < banks[height].section_boundaries[0]) {
            return Expr::Var(left);
        }

        const Expr* left_expr = reconstruct(height - 1, left);

        if (index < banks[height].section_boundaries[1]) {
            //std::cout << "height " << height << ", picking NOT" << std::endl;
            return Expr::Not(left_expr);
        }

        uint32_t right = banks[height].get_right(index);
        const Expr* right_expr = reconstruct(height - 1, right);

        if (index < banks[height].section_boundaries[2]) {
            //std::cout << "height " << height << ", picking AND" << std::endl;
            return Expr::And(left_expr, right_expr);
        }

        if (index < banks[height].section_boundaries[3]) {
            //std::cout << "height " << height << ", picking OR" << std::endl;
            //std::cout << "(index: " << index << ", boundary: " << banks[height].section_boundaries[3] << std::endl;
            return Expr::Or(left_expr, right_expr);
        }

        if (index < banks[height].section_boundaries[4]) {
            //std::cout << "height " << height << ", picking XOR" << std::endl;
            return Expr::Xor(left_expr, right_expr);
        }

        // Index out of range.
        assert(false);
        return nullptr;
    }

    const Expr* synthesize(std::ostream &out) {
        for (int32_t height = 0; height <= spec.sol_height; height++) {
            std::cerr << "synthesizing height " << height << std::endl;

            banks.push_back(Bank(max_distinct_terms));

            // Insert variables with the current height.
            for (uint32_t i = 0; i < spec.num_vars; i++) {
                if (spec.var_heights[i] == height) {
                    banks[height].insert_unary(spec.var_values[i], i);
                }
            }
            banks[height].end_section();

            if (height == 0) {
                // Skip NOT, AND, OR, and XOR.
                for (uint32_t i = 0; i < 4; i++) {
                    banks[height].end_section();
                }
                continue;
            }

            // Did we find a term matching the desired output?
            bool found = false;

            // Synthesize NOT terms.
            for (uint32_t left = 0; !found && left < banks[height - 1].size(); left++) {
                uint32_t result = ~banks[height - 1].get_results(left);
                //update the result to only have the relevant bits
                result = result & result_mask;
                banks[height].insert_unary(result, left);
                if (result == spec.sol_result) {
                    found = true;
                }
            }
            banks[height].end_section();

            // Synthesize AND terms.
            for (uint32_t left = 0; !found && left < banks[height - 1].size(); left++) {
                for (uint32_t right = 0; !found && right <= left; right++) {
                    uint32_t result = banks[height - 1].get_results(left)
                        & banks[height - 1].get_results(right);
                    //update the result to only have the relevant bits
                    result = result & result_mask;
                    banks[height].insert_binary(result, left, right);
                    if (result == spec.sol_result) {
                        found = true;
                    }
                }
            }
            banks[height].end_section();

            // Synthesize OR terms.
            for (uint32_t left = 0; !found && left < banks[height - 1].size(); left++) {
                for (uint32_t right = 0; !found && right <= left; right++) {
                    uint32_t result = banks[height - 1].get_results(left)
                        | banks[height - 1].get_results(right);
                    //update the result to only have the relevant bits
                    result = result & result_mask;
                    banks[height].insert_binary(result, left, right);
                    if (result == spec.sol_result) {
                        found = true;
                    }
                }
            }
            banks[height].end_section();

            // Synthesize XOR terms.
            for (uint32_t left = 0; !found && left < banks[height - 1].size(); left++) {
                for (uint32_t right = 0; !found && right <= left; right++) {
                    uint32_t result = banks[height - 1].get_results(left)
                        ^ banks[height - 1].get_results(right);
                    //update the result to only have the relevant bits
                    result = result & result_mask;
                    banks[height].insert_binary(result, left, right);
                    if (result == spec.sol_result) {
                        found = true;
                    }
                }
            }
            banks[height].end_section();

            if (found) {
                const Expr* expr = reconstruct(height, banks[height].size() - 1);

                // This expression might not be the desired height, but we can
                // AND it with itself and NOT it twice until it's deep enough.
                uint32_t remaining_height = spec.sol_height - height;
                if (remaining_height % 2 == 1) {
                    expr = Expr::And(expr, expr);
                    remaining_height--;
                }
                while (remaining_height > 0) {
                    expr = Expr::Not(Expr::Not(expr));
                    remaining_height -= 2;
                }

                //std::cerr << "validating solution" << std::endl;
                spec.validate(expr);

                out << "bank size: " << banks[height].size() << std::endl;
                return expr;
            }
        }

        out << "bank size: " << banks[spec.sol_height].size() << std::endl;
        return nullptr;
    }

    void print_banks(std::ostream &out) {
        for (size_t height = 0; height < banks.size(); height++) {
            out << "height " << height << ":\n";
            for (size_t i = 0; i < banks[height].size(); i++) {
                out << "\t";
                reconstruct(height, i)->print(out, &spec.var_names);
                out << "\n";
            }
        }
    }
};

int main(void) {
    ofstream outputFile;
    outputFile.open("output.txt");

    std::string dir_path = "./inputs/";
    //std::string dir_path = "./our_inputs/";
    std::string current_path;
    for (const auto & entry : fs::directory_iterator(dir_path)) {
        current_path = entry.path().string();

        outputFile << current_path << std::endl;

        Spec spec = Parser::parseInput(current_path);
        //Spec spec = Parser::parseTruthTableInput(current_path);

	/*std::cout<<"Number of variables: "<<spec.num_vars << ", number of examples: "<<spec.num_examples << std::endl;
	for (int i = 0; i < spec.var_names.size(); i++)
		std::cout << spec.var_names[i] << " with height " << spec.var_heights[i] << ", ";
	std::cout << std::endl;*/
	//std::cout<<"I/O examples:"<<std::endl;

	/*for (int i = 0; i < spec.var_names.size(); i++)
		std::cout << spec.var_values[i] << ", ";
	std::cout << std::endl;
	std::cout << spec.sol_result << std::endl;*/


        if (spec.num_vars > 7) {
            outputFile << "Skipping this one because it has too many (" << spec.num_vars << ") variables" << std::endl << std::endl;
            continue;
        }

        //std::cout << "look at: " << (1L << spec.num_examples) << std::endl;
        //std::cout << "look at: " << (1ULL << 32) << std::endl;

        const Expr* expr = nullptr;
        // The i'th element specifies the values of the i'th variable,
        // where the j'th bit of that integer is the variable's value in example j.
        // used to update spec.var_values
        std::vector<uint32_t> updated_var_vals(spec.num_vars);
        // The i'th bit is the desired output in example i.
        // used to update spec.sol_result
        uint32_t updated_sol_result;
        int i=0;
        while(true) {
            Synthesizer synthesizer(spec);
            cout<<"synthesizing"<<std::endl;
            expr = synthesizer.synthesize(outputFile);
            cout<<"done synthesizing"<<std::endl;
            if(expr==nullptr) break;
            int counterExample = spec.counterexample(expr);
            if(counterExample == -1) break;
            outputFile << "Candidate (counterexample found "<<counterExample<<"): ";
            expr->print(outputFile, &spec.var_names);
            outputFile << std::endl;

            // Update the spec
            int r=i%32;
            for(uint32_t j=0; j<spec.var_values.size(); j++) {
                updated_var_vals[j] = spec.var_values[j] & ~(1<<r);
                updated_var_vals[j] |= (spec.all_inputs[counterExample][j]?1:0)<<r;
            }
            updated_sol_result = spec.sol_result & ~(1<<r);
            updated_sol_result |= (spec.all_sols[counterExample]?1:0)<<r;
            spec.updateIOExamples(updated_var_vals,updated_sol_result);
            
            cout<<"Iteration "<<i<<" "<<counterExample<<std::endl;
            i++;
        }
        if (expr == nullptr) {
            outputFile << "no solution" << std::endl;
        } else {
            outputFile << "solution: ";
            expr->print(outputFile, &spec.var_names);
            outputFile << std::endl;
        }

        outputFile << std::endl;

        //synthesizer.print_banks(std::cout);
    }

    outputFile.close();
}
