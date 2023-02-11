#ifndef SPEC_H
#define SPEC_H

#include <bitset>
#include <cstdint>
#include <iostream>
#include <vector>

class Spec {
public:
    // Number of variables.
    uint32_t num_vars;

    // Number of input/output examples.
    uint32_t num_examples;

    // The i'th element is the name of the i'th variable.
    std::vector<std::string> var_names;

    // The i'th element is the depth of the i'th variable.
    std::vector<uint32_t> var_depths;

    // The i'th element specifies the values of the i'th variable,
    // where the j'th bit of that integer is the variable's value in example j.
    std::vector<uint32_t> var_values;

    // The i'th bit is the desired output in example i.
    uint32_t sol_result;

    // The depth of the solution circuit.
    uint32_t sol_depth;

    Spec(
        uint32_t num_vars,
        uint32_t num_examples,
        std::vector<std::string> var_names,
        std::vector<uint32_t> var_depths,
        std::vector<uint32_t> var_values,
        uint32_t sol_result,
        uint32_t sol_depth
    ) :
        num_vars(num_vars),
        num_examples(num_examples),
        var_names(var_names),
        var_depths(var_depths),
        var_values(var_values),
        sol_result(sol_result),
        sol_depth(sol_depth) {}

    friend std::ostream& operator<< (std::ostream &out, const Spec &spec) {
        out << "num_vars: " << spec.num_vars
            << ", num_examples: " << spec.num_examples;

        out << ", var_names:";
        for (auto name : spec.var_names) {
            out << " " << name;
        }

        out << ", var_depths:";
        for (auto depth : spec.var_depths) {
            out << " " << depth;
        }

        out << ", var_values:";
        for (auto value : spec.var_values) {
            out << " " << std::bitset<32>(value);
        }

        out
            << ", sol_result: " << std::bitset<32>(spec.sol_result)
            << ", sol_depth: " << spec.sol_depth;

        return out;
    }
};

#endif
