// Run the synthesizer variant of your choice on all of the SyGuS cryptography benchmarks
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

int main(void) {
    std::cerr << "Synthesizer variant: " << VARIANT_DESCRIPTION << std::endl;

    ofstream outputFile;
    outputFile.open("synth_cpu_test.txt");

    std::string dir_path = "./inputs/";
    std::string current_path;
    for (const auto & entry : fs::directory_iterator(dir_path)) {
        current_path = entry.path().string();

        outputFile << current_path << std::endl;
        std::cout << current_path << std::endl;

        Spec spec = Parser::parseInput(current_path);

        /*if (spec.num_vars == 5) {
            for (int i = 0; i < spec.num_examples; i++) {
                for (int j = 0; j < spec.num_vars; j++) {
                    outputFile << ((spec.var_values[j] >> i) & 1);
                }
                outputFile << endl;
            }
            outputFile << endl;
        }*/


        if (spec.num_vars > 5) {
            outputFile << "Skipping this one because it has too many (" << spec.num_vars << ") variables" << std::endl << std::endl;
            continue;
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
            cout<<"synthesizing"<<std::endl;
            //expr = synthesizer.synthesize(outputFile);
            expr = synthesizer.synthesize();
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

            const Expr* constant_height_solution = expr->with_constant_height(
                spec.sol_height, spec.var_heights);

            outputFile << "constant height solution: ";
            constant_height_solution->print(outputFile, &spec.var_names);
            outputFile << std::endl;

            spec.validate(constant_height_solution);
        }

        outputFile << std::endl;
    }

    outputFile.close();
}
