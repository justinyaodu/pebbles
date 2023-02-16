#ifndef SPEC_H
#define SPEC_H

#include <bitset>
#include <cstdint>
#include <iostream>
#include <vector>

#include "expr.hpp"

class Spec {
public:
    // Number of variables.
    const uint32_t num_vars;

    // Number of input/output examples.
    const uint32_t num_examples;

    // The i'th element is the name of the i'th variable.
    const std::vector<std::string> var_names;

    // The i'th element is the height of the i'th variable.
    const std::vector<int32_t> var_heights;

    // The i'th element specifies the values of the i'th variable,
    // where the j'th bit of that integer is the variable's value in example j.
    const std::vector<uint32_t> var_values;

    // The i'th bit is the desired output in example i.
    const uint32_t sol_result;

    // The height of the solution circuit.
    const int32_t sol_height;

    Spec(
        uint32_t num_vars,
        uint32_t num_examples,
        std::vector<std::string> var_names,
        std::vector<int32_t> var_heights,
        std::vector<uint32_t> var_values,
        uint32_t sol_result,
        int32_t sol_height
    ) :
        num_vars(num_vars),
        num_examples(num_examples),
        var_names(var_names),
        var_heights(var_heights),
        var_values(var_values),
        sol_result(sol_result),
        sol_height(sol_height) {}

    void validate(const Expr* solution) {
        solution->assert_constant_height(sol_height, var_heights);
        for (uint32_t example = 0; example < num_examples; example++) {
            std::vector<bool> vars;
            for (uint32_t var = 0; var < num_vars; var++) {
                vars.push_back((var_values[var] >> example) & 1);
            }
            bool expected = (sol_result >> example) & 1;
            assert(solution->eval(vars) == expected);
        }
    }

    friend std::ostream& operator<< (std::ostream &out, const Spec &spec) {
        out << "num_vars: " << spec.num_vars
            << ", num_examples: " << spec.num_examples;

        out << ", var_names:";
        for (auto name : spec.var_names) {
            out << " " << name;
        }

        out << ", var_heights:";
        for (auto height : spec.var_heights) {
            out << " " << height;
        }

        out << ", var_values:";
        for (auto value : spec.var_values) {
            out << " " << std::bitset<32>(value);
        }

        out
            << ", sol_result: " << std::bitset<32>(spec.sol_result)
            << ", sol_height: " << spec.sol_height;

        return out;
    }
};

#endif
