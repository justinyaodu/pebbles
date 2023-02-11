#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
using namespace std;

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

    cout<<" "<<expr;

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
    string expr = "(xor  LN29(xor  LN3(and  LN19 LN20 ) ) )";
    vector<string> vars;
    vars.push_back("LN3");
    vars.push_back("LN19");
    vars.push_back("LN20");
    vars.push_back("LN29");
    vector<bool> vals;
    vals.push_back(true);
    vals.push_back(true);
    vals.push_back(false);
    vals.push_back(false);

    truthTable(expr, vars, vals);
}

