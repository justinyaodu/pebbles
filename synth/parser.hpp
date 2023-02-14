#ifndef PARSER_H
#define PARSER_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <stack>
#include <bitset>
using namespace std;

#include "spec.hpp"

class Parser {
private:
    static vector<uint32_t> getVarValues(uint32_t numVariables, uint32_t numExamples);
public:
    static Spec parseInput(string inputFileName);
};

#endif