#include <random>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <functional>
#include <cassert>
#include <bitset>
using namespace std;

void makeInputFileFromFunction(string outFile, int maxHeight, int numVariables, vector<string> varNames, vector<int> varHeights, function<uint32_t(vector<bool>)> evalFunction) {
    ofstream outputFile;
    outputFile.open(outFile);

    outputFile << "max-height:" << endl;
    outputFile << maxHeight << endl;
    outputFile << "done" << endl;
    outputFile << endl;

    outputFile << "variables:" << endl;
    for (int i = 0; i < numVariables; i++) {
        outputFile << varNames[i] << " " << varHeights[i] << endl;
    }
    outputFile << "done" << endl;
    outputFile << endl;

    outputFile << "input/output:" << endl;
    for (uint32_t i = 0; i < (uint32_t) pow(2,numVariables); i++) {
        vector<bool> inputs;
        for(int j=0; j< numVariables; j++) {
            inputs.push_back((i>>j)&1);
        }
        for (bool inputBit : inputs) {
            outputFile << (int) inputBit;
        }
        outputFile << " " << evalFunction(inputs) << endl;
    }
    outputFile << "done" << endl;

    outputFile.close();
}

// Returns either or 0 or 1 (equal chance of each)
uint32_t randomValue(vector<bool> varVals) {
    uint32_t retVal = (int) (rand() % 2);
    assert(retVal == 0 || retVal == 1);
    return retVal;
}

// Returns 1 if there's an odd number of trues and 0 if there's an even number
uint32_t bigXOR(vector<bool> varVals) {
    bool retVal = 0;
    for (bool val : varVals) {
        retVal ^= val;
    }
    return (uint32_t) retVal;
}

void makeInputFileFromRandom(string outFile, int maxHeight, int numVariables, vector<string> varNames, vector<int> varHeights) {
    srand(time(NULL));
    makeInputFileFromFunction(outFile,maxHeight,numVariables,varNames,varHeights,randomValue);
}

int main() {
    makeInputFileFromRandom("./our_inputs/input5.txt",
        5,
        3,
        vector<string>{"var1","var2","var3"},
        vector<int>{0,1,1});
    return 0;
}