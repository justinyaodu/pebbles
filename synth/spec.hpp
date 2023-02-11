#ifndef SPEC_H
#define SPEC_H

#include <cstdint>
#include <vector>

class Spec {
public:
    // Number of variables.
    uint32_t num_vars;

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
        std::vector<uint32_t> var_depths,
        std::vector<uint32_t> var_values,
        uint32_t sol_result,
        uint32_t sol_depth
    ) :
        num_vars(num_vars),
        var_depths(var_depths),
        var_values(var_values),
        sol_result(sol_result),
        sol_depth(sol_depth) {}
};

#endif
