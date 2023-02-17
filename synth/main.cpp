#include <iostream>
#include <string>
#include <vector>

#include "spec.hpp"

#ifndef SYNTH_VARIANT
#error "SYNTH_VARIANT must be defined. See the Makefile."
#elif SYNTH_VARIANT == 1
#include "synth_cpu_st.hpp"
#elif SYNTH_VARIANT == 2
#include "synth_cpu_mt.hpp"
#else
#error "Unsupported SYNTH_VARIANT."
#endif

int main(void) {
    /*
    Spec spec(
        4,
        16,
        std::vector<std::string> {"a", "b", "c", "d"},
        std::vector<int32_t> {0, 0, 0, 0},
        std::vector<uint32_t> {
            // 0b00000000000000001111111111111111,
            0b0000000011111111, // 0000000011111111,
            0b0000111100001111, //0000111100001111,
            0b0011001100110011, //0011001100110011,
            0b0101010101010101 //0101010101010101
        },
        0b1011110100111000,
        32
    );
    */

    /*
    Spec spec(
        5,
        32,
        std::vector<std::string> {"a", "b", "c", "d", "e"},
        std::vector<int32_t> {0, 0, 0, 0, 4},
        std::vector<uint32_t> {
            0b00000000000000001111111111111111,
            0b00000000111111110000000011111111,
            0b00001111000011110000111100001111,
            0b00110011001100110011001100110011,
            0b01010101010101010101010101010101
        },
        0b00111001110011100000000011001110,
        32
    );
    */

    Spec spec(
        5,
        24,
        std::vector<std::string> {"a", "b", "c", "d", "e"},
        std::vector<int32_t> {0, 0, 0, 0, 4},
        std::vector<uint32_t> {
            0b000000000000000011111111,
            0b000000001111111100000000,
            0b000011110000111100001111,
            0b001100110011001100110011,
            0b010101010101010101010101,
        },
        0b001110011100111000000000,
        32
    );

    Synthesizer synthesizer(spec);

    const Expr* solution = synthesizer.synthesize();

    if (solution == nullptr) {
        std::cout << "no solution" << std::endl;
    } else {
        std::cout << "solution: ";
        solution->print(std::cout, &spec.var_names);
        std::cout << std::endl;

        const Expr* constant_height_solution = solution->with_constant_height(
                spec.sol_height, spec.var_heights);

        std::cout << "constant height solution: ";
        constant_height_solution->print(std::cout, &spec.var_names);
        std::cout << std::endl;

        spec.validate(constant_height_solution);
    }
}
