#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <stack>
#include <bitset>
using namespace std;

#include "spec.hpp"
#include "parser.hpp"

/**
 * Java styled ReplaceAll, taken from StackOverflow
*/
std::string ReplaceAll(std::string str, const std::string& from, const std::string& to) {
    size_t start_pos = 0;
    while((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
    }
    return str;
}

/**
 * Java styled pop for a stack; pops and returns the value
*/
bool popVal(stack<bool> &st) {
    bool val=st.top();
    st.pop();
    return val;
}

bool evalExpr(string expr, vector<string> names, vector<bool> vals) {
    // replace all variables with their values
    for(int i=0; i<names.size(); i++) {
        expr = ReplaceAll(expr, names[i], vals[i]?"T":"F");
    }
    // replace all gates with single letters so it's easier to parse
    expr = ReplaceAll(expr, "xor", "^");
    expr = ReplaceAll(expr, "or", "|");
    expr = ReplaceAll(expr, "and", "&");
    expr = ReplaceAll(expr, "not", "!");

    //cout<<" "<<expr;

    // use a stack to track operators like in postfix expressions
    stack<bool> st;

    // push any boolean values to the stack. once it finds an operator,
    // pop the necessary number of operators and push that to the stack
    for(int i=expr.length()-1; i>=0; i--) {
        if(expr[i]=='T') st.push(true);
        if(expr[i]=='F') st.push(false);
        if(expr[i]=='&') st.push(popVal(st)&popVal(st));
        if(expr[i]=='|') st.push(popVal(st)|popVal(st));
        if(expr[i]=='^') st.push(popVal(st)^popVal(st));
        if(expr[i]=='!') st.push(!popVal(st));
    }

    // should only ever be one value left, and that's the return
    return st.top();
}

int power(int x, int y) {
    return (y==0)?1:x*power(x, y-1);
}

vector<bool> truthTable(string expr, vector<string> names, vector<bool> vals) {
    // set up the vector to return
    vector<bool> retVal;
    // will be 2^n options
    for(int i=0; i<power(2, names.size()); i++) {
        for(int j=0; j<vals.size(); j++) {
            // get the jth bit
            vals[j]=(i>>j)&1;
            cout << vals[j];
        }
        retVal.push_back(evalExpr(expr, names, vals));
        cout<<" "<<retVal.back()<<"\n";
    }
    return retVal;
}

uint32_t truthTableIntResult(string expr, vector<string> names, vector<bool> vals) {
    // set up the integer to return
    uint32_t retVal = 0;
    // will be 2^n options
    for(int i=0; i<power(2, names.size()); i++) {
        for(int j=0; j<vals.size(); j++) {
            // get the jth bit
            vals[j]=(i>>j)&1;
        }
        retVal = (retVal << 1) | evalExpr(expr, names, vals);
    }
    return retVal;
}

vector<uint32_t> Parser::getVarValues(uint32_t numVariables, uint32_t numExamples) {
    uint32_t currVar;
    vector<uint32_t> varValues;
    for (int i = 0; i < numVariables; i++) {
        currVar = 0;
        for (int j = 0; j < numExamples; j++) {
            currVar = (currVar << 1) | ((j >> i) % 2 == 0 ? 0 : 1);
        }
        varValues.push_back(currVar);
    }
    return varValues;
}
Spec Parser::parseInput(string inputFileName) {
    uint32_t numVariables = 0;
    uint32_t maxDepth = 0;//e.g. this will be 4 for the D5 files since the grammar has "Start" as a sort of 0 level depth
    vector<uint32_t> var_depths;//depths range from 0 to maxDepth, represent the "weight" of the variable in the tree
    vector<string> var_names;
    uint32_t sol_result;
    uint32_t num_examples;

    //open SyGuS-formatted input file
    ifstream inputFile;
    inputFile.open("input.sl");
    int lineNumber = 1;
    string line;

    //where we are in the file:
    bool finishedGrammar = false;
    bool originalCircuitLine = false;
    int depth = 0;

    //specification of the original non-constant circuit
    string origCir;

    //temp variables:
    string varName;

    while (getline(inputFile, line))
    {
        if (originalCircuitLine)
        {
            originalCircuitLine = false;
            //line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
            origCir = line;
        } else if (line.find("define-fun origCir") != string::npos)
        {
            originalCircuitLine = true;//next time
        } else if (lineNumber > 10 && !finishedGrammar)
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
    num_examples = power(2,numVariables);
    inputFile.close();

    //Flip depths to be "weight"s instead
    for (int i = 0; i < numVariables; i++)
    {
        var_depths[i] = maxDepth - var_depths[i];
    }

    /*
    //print out all the variables
    for (int i = 0; i < numVariables; i++)
    {
        cout << var_names[i] << ": depth " << var_depths[i] << endl;
    }*/

    vector<bool> vals;
    for (int i = 0; i < numVariables; i++) {
        vals.push_back(true);
    }
    //truthTable(origCir, var_names, vals);
    sol_result = truthTableIntResult(origCir, var_names, vals);

    return Spec(numVariables, num_examples, var_names, var_depths, getVarValues(numVariables, num_examples), sol_result, maxDepth+1);
}


int testParser()
{
    vector<string> vars;
    vars.push_back("b1");
    vars.push_back("b2");
    vars.push_back("b3");
    vars.push_back("b4");
    vector<bool> vals;
    for (int i = 0; i < 4; i++) {
        vals.push_back(true);
    }
    cout << Parser::parseInput("input.sl") << endl;
    /*<uint32_t> temp = Parser::getVarValues(4,16);
    for (int i = 0; i < temp.size(); i++) {
        cout << bitset<16>(temp[i]) << endl;
    }*/
    //truthTable("(not (and (not b1) ) b2 )", vars, vals);
}

