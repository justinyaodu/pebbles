// Run the synthesizer variant of your choice on one file from the SyGuS cryptography benchmarks
// (that have 7 or fewer variables)

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

#ifndef SYNTH_VARIANT
#error "SYNTH_VARIANT must be defined. See the Makefile."
#elif SYNTH_VARIANT == 1
#include "synth_cpu_st.hpp"
#define VARIANT_DESCRIPTION "CPU, single threaded"
#elif SYNTH_VARIANT == 2
#include "synth_cpu_mt.hpp"
#define VARIANT_DESCRIPTION "CPU, multi-threaded"
#elif SYNTH_VARIANT == 3
#include "synth_gpu.cu"
#define VARIANT_DESCRIPTION "GPU"
#else
#error "Unsupported SYNTH_VARIANT."
#endif

int main(int argc, char *argv[]) {
    std::cerr << "Synthesizer variant: " << VARIANT_DESCRIPTION << std::endl;

    if (argc < 2) {
        std::cout << "Please specify an input file" << std::endl;
        return 1;
    }
    std::string file_path = argv[1];
    std::cerr << file_path << std::endl;

    Spec spec = Parser::parseInput(file_path);

    if (spec.num_vars > 8) {
        std::cout << "Skipping this one because it has too many (" << spec.num_vars << ") variables" << std::endl << std::endl;
        return 1;
    }

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
        //expr = synthesizer.synthesize(outputFile);
        expr = synthesizer.synthesize();
        if(expr==nullptr) break;
        int counterExample = spec.advanceCEGISIteration(expr);
        if(counterExample == -1) break;
        i++;
    }
    if (expr == nullptr) {
        std::cout << "no solution found in " << i << " iterations"<<std::endl;
    } else {
        std::cout << "solution found in " << i << " iterations: ";
        expr->print(std::cout, &spec.var_names);
        std::cout << std::endl;

        const Expr* constant_height_solution = expr->with_constant_height(
            spec.sol_height, spec.var_heights);

        std::cout << "constant height solution: ";
        constant_height_solution->print(std::cout, &spec.var_names);
        std::cout << std::endl;

        spec.validate(constant_height_solution);
    }
}
