#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
using namespace std;

int main()
{
    //open SyGuS-formatted input file
    ifstream inputFile;
    inputFile.open("input.sl");
    int lineNumber = 1;
    string line;
    bool finishedGrammar = false;
    int depth = 0;
    string varName;
    vector<uint32_t> var_depths;
    vector<string> var_names;
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
    inputFile.close();

    //print out all the variables
    for (int i = 0; i < var_names.size(); i++) {
        cout << var_names[i] << ": depth " << var_depths[i] << endl;
    }
}