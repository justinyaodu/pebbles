#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
using namespace std;

int main()
{
    //What we're getting from the input file (todo: create a spec from this)
    uint32_t numVariables = 0;
    uint32_t maxDepth = 0;//e.g. this will be 4 for the D5 files since the grammar has "Start" as a sort of 0 level depth
    vector<uint32_t> var_depths;//depths range from 0 to maxDepth, represent the "weight" of the variable in the tree
    vector<string> var_names;

    //open SyGuS-formatted input file
    ifstream inputFile;
    inputFile.open("input.sl");
    int lineNumber = 1;
    string line;
    bool finishedGrammar = false;
    int depth = 0;
    string varName;
    while (getline(inputFile, line))
    {
        if (lineNumber > 10 && !finishedGrammar)
        {
            //we are in the grammar-defining portion
            if (line == ")")
            {
                finishedGrammar = true;
            } else if (line.find("(depth") != string::npos)
            {
                depth++;
            } else if (line.find("(") == string::npos && line.find(")") == string::npos)
            {
                line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
                var_depths.push_back(depth);
                var_names.push_back(line);
            }
        }
        lineNumber++;
    }
    maxDepth = depth;
    numVariables = var_names.size();
    inputFile.close();

    //Flip depths to be "weight"s instead
    for (int i = 0; i < numVariables; i++)
    {
        var_depths[i] = maxDepth - var_depths[i];
    }

    //print out all the variables
    for (int i = 0; i < numVariables; i++)
    {
        cout << var_names[i] << ": depth " << var_depths[i] << endl;
    }
}