#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <stack>
#include <bitset>
#include <random>
using namespace std;

#include "spec.hpp"
#include "parser.hpp"

enum FileSection { Height, Variables, InputOutput, None };

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
    for(uint32_t i=0; i<names.size(); i++) {
        expr = ReplaceAll(expr, names[i], vals[i]?"T":"F");
    }
    // replace all gates with single letters so it's easier to parse
    expr = ReplaceAll(expr, "false", "F");
    expr = ReplaceAll(expr, "true", "T");
    expr = ReplaceAll(expr, "xor", "^");
    expr = ReplaceAll(expr, "or", "|");
    expr = ReplaceAll(expr, "and", "&");
    expr = ReplaceAll(expr, "not", "!");

    // cout<<" "<<expr<<std::endl;

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
        for(uint32_t j=0; j<vals.size(); j++) {
            // get the jth bit
            vals[j]=(i>>j)&1;
        }
        retVal.push_back(evalExpr(expr, names, vals));
    }
    return retVal;
}

uint32_t truthTableIntResult(string expr, vector<string> names, vector<bool> vals) {
    // set up the integer to return
    uint32_t retVal = 0;
    // will be 2^n options
    for(int i=0; i<power(2, names.size()); i++) {
        for(uint32_t j=0; j<vals.size(); j++) {
            // get the jth bit
            vals[j]=(i>>j)&1;
        }
        retVal = (retVal << 1) | evalExpr(expr, names, vals);
    }
    return retVal;
}

uint32_t min(int a, int b) {
    return (a<b)?a:b;
}

// Pull a random set of 32 input/output examples from the entire truth table (truth table could be incomplete)
// Returns an integer holding the output values for the selected examples (ith bit has the output for the ith example)
uint32_t truthTableWithVecFromTruthTable(const vector<bool> &all_sols, const vector<vector<bool>> &all_inputs, vector<uint32_t> &vals) {
    // this is what we'll return at the end
    uint32_t outVals = 0;

    // We want to select 32 indices into all_sols/all_inputs for our set of examples
    // To do this, start by making a vector containing all possible indices into those vectors (0 through length-1)
	vector<uint32_t> indices(all_sols.size());
	for (uint32_t i = 0; i < all_sols.size(); i++) {
		indices[i]=i;
	}

    // Now we shuffle the indices
    // Before the first 32 entries in indices were 0-32. After the first 32 entries will be some random indices
    // (If we have fewer than 5 variables, we'll just shuffle the < 32 indices into some other order, 
    // and we'll end up taking all of them anyway)
	auto rng = default_random_engine{};
    rng.seed(time(NULL));
	shuffle(begin(indices), end(indices), rng);

	vector<bool> currExample;
    // Ideally we'd want to have 2^(#varaibles) examples, one for each possible set of inputs, but we're capped at 32
	uint32_t numExamples = min(32, power(2, all_inputs[0].size()));
	for (uint32_t i = 0; i < numExamples; i++) {
		currExample = all_inputs[indices[i]];
		for (uint32_t j = 0; j < currExample.size(); j++) {
            // append the value of variable j in example i to the integer holding the values of variable j
			vals[j] = (vals[j] << 1) | currExample[j];
		}
        // append the output value in example i to the integer holding all of the output values
		outVals = (outVals << 1) | all_sols[indices[i]];
	}
	return outVals;
}

uint32_t truthTableWithVec(string expr, vector<string> names, vector<uint32_t> &vals) {
    uint32_t retVal = 0;
    for(uint32_t i=0; i<min(power(2, names.size()), 32); i++) {
        int x=((1+(power(2, names.size())/32))*i)%power(2, names.size());
        if(x == 2) x = 1;
        vector<bool> input;
        for(uint32_t j=0; j<names.size(); j++) {
            vals[j] = (vals[j] << 1) | ((x>>j)&1);
            input.push_back((x>>j)&1);
        }
        retVal = (retVal << 1) | evalExpr(expr, names, input);
    }
    return retVal;
}

vector<bool> truthTableFull(string expr, vector<string> names, vector<vector<bool>> &vals) {
    vector<bool> retVal;
    for(int i=0; i<power(2, names.size()); i++) {
        vector<bool> newVec;
        vals.push_back(newVec);
        for(uint32_t j=0; j<names.size(); j++) {
            vals[i].push_back((i>>j)&1);
        }
        retVal.push_back(evalExpr(expr, names, vals[i]));
    }
    return retVal;
}

vector<uint32_t> Parser::getVarValues(uint32_t numVariables, uint32_t numExamples) {
    uint32_t currVar;
    vector<uint32_t> varValues;
    for (uint32_t i = 0; i < numVariables; i++) {
        currVar = 0;
        for (uint32_t j = 0; j < numExamples; j++) {
            currVar = (currVar << 1) | ((j >> i) % 2 == 0 ? 0 : 1);
        }
        varValues.push_back(currVar);
    }
    return varValues;
}

Spec Parser::parseTruthTableInput(string inputFileName) {
    uint32_t numVariables = 0;
    int32_t maxHeight = 0;//e.g. this will be 4 for the D5 files since the grammar has "Start" as a sort of 0 level height
    vector<int32_t> var_heights;//heights range from 0 to maxHeight, represent the "weight" of the variable in the tree
    vector<string> var_names;
    uint32_t num_examples;
    vector<vector<bool>> all_inputs;
    vector<bool> full_sol;

    //open truth table-formatted input file
    ifstream inputFile;
    inputFile.open(inputFileName);
    int spaceAt;
    string inputs;
    // for a particular example
    string line;

    //where we are in the file:
    FileSection section = None;

    while (getline(inputFile, line)) {
        if (line.find("done") != string::npos) {
            section = None;
        } else if (section == Height) {
            maxHeight = stoi(line);
        } else if (section == Variables) {
            spaceAt = line.find(' ');
            var_names.push_back(line.substr(0,spaceAt));
            var_heights.push_back(stoi(line.substr(spaceAt+1)));
        } else if (section == InputOutput) {
            spaceAt = line.find(' ');
            //output
            full_sol.push_back(line.at(spaceAt+1)-'0');
            //input
            inputs = line.substr(0,spaceAt);
	    //cout << inputs << " " << line.at(spaceAt+1) << endl;
            vector<bool> inputVals(var_names.size());
            for (uint32_t i = 0; i < inputs.length(); i++) {
		        char c = inputs.at(i);
                inputVals[i] = c-'0';
            }
	    /*for (bool b : inputVals) {
		    cout << b << ", ";
	    }
	    cout << endl;*/
            all_inputs.push_back(inputVals);
        } else if (line.find("max-height:") != string::npos) {
            section = Height;
        } else if (line.find("variables:") != string::npos) {
            section = Variables;
        } else if (line.find("input/output:") != string::npos) {
            section = InputOutput;
        }
    }
    numVariables = var_names.size();
    num_examples = power(2,numVariables);
    if(num_examples>32) num_examples=32;
    inputFile.close();

    /*for (uint32_t i = 0; i < all_inputs.size(); i++) {
	    for (uint32_t j = 0; j < all_inputs[i].size(); j++) {
		    cout << all_inputs[i][j] << ", ";
	     }
	    cout << "output: " << full_sol[i] << endl;
    }*/

    return Spec(numVariables, 
                num_examples, 
                var_names, 
                var_heights,
                maxHeight,
                all_inputs,
                full_sol);
}

Spec Parser::parseInput(string inputFileName) {
    uint32_t numVariables = 0;
    int32_t maxDepth = 0;//e.g. this will be 4 for the D5 files since the grammar has "Start" as a sort of 0 level depth
    vector<int32_t> var_depths;//depths range from 0 to maxDepth, represent the "weight" of the variable in the tree
    vector<string> var_names;
    uint32_t num_examples;

    //open SyGuS-formatted input file
    ifstream inputFile;
    inputFile.open(inputFileName);
    int lineNumber = 1;
    string line;

    //where we are in the file:
    bool startedGrammar = false;
    bool finishedGrammar = false;
    bool originalCircuitLine = false;
    int depth = 0;

    //specification of the original non-constant circuit
    string origCir;

    //temp variables:
    string varName;

    while (getline(inputFile, line))
    {
        if (line.size() > 0 && line.at(0) == ';')
        {
            //cout << "in a comment" << endl;
            // this line must be a comment
        } else if (originalCircuitLine)
        {
            originalCircuitLine = false;
            //line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
            origCir = line;
            //cout << origCir << endl;
        } else if (line.find("define-fun origCir") != string::npos || line.find("define-fun Spec") != string::npos)
        {
            //cout << "about to get the original circuit" << endl;
            originalCircuitLine = true;//next time
        } else if (line.find("synth-fun") != string::npos)
        {
            //cout << "about to get the grammar" << endl;
            startedGrammar = true;//next time
        } if (startedGrammar && !finishedGrammar)
        {
            //we are in the grammar-defining portion
            if (line == ")")
            {
                //cout << "finished the grammar" << endl;
                finishedGrammar = true;
            } else if (line.find("(depth") != string::npos)
            {
                //cout << "depth going up!" << endl;
                depth++;
            } else if (line.find("(") == string::npos && line.find(")") == string::npos)
            {
                //cout << "new variable" << endl;
                line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
                var_depths.push_back(depth);
                var_names.push_back(line);
            } else {
                //cout << "something else" << endl;
            }
        }
        lineNumber++;
    }
    maxDepth = depth;
    //maxDepth = depth+5;
    numVariables = var_names.size();
    num_examples = power(2,numVariables);
    if(num_examples>32) num_examples=32;
    inputFile.close();

    //Flip depths to be "height"s instead
    for (uint32_t i = 0; i < numVariables; i++)
    {
        var_depths[i] = maxDepth - var_depths[i];
        //var_depths[i] = 0;
    }

    if (numVariables > 31) {
        // we can't even parse this many. abort
        cout << "Abandoning this spec because it has too many (" << numVariables << ") variables" << endl;
        return Spec(numVariables, 
                0, 
                var_names, 
                var_depths, 
                maxDepth,
                vector<vector<bool>>{},
                vector<bool>{});
    }

    //sol_result = truthTableWithVec(origCir, var_names, vals);

    cout<<"spec making"<<std::endl;

    vector<vector<bool>> all_inputs;
    cout<<origCir<<endl;
    for (int i = 0; i < var_names.size(); i++) {
        cout<<var_names[i]<<endl;
    }
    vector<bool> full_sol = truthTableFull(origCir, var_names, all_inputs);

    cout<<"spec made"<<std::endl;

    return Spec(numVariables, 
                1, // could be num_examples
                var_names, 
                var_depths,
                maxDepth,
                all_inputs,
                full_sol);
}


void testParser()
{
    vector<string> vars;
    vars.push_back("b1");
    vars.push_back("b2");
    vars.push_back("b3");
    vars.push_back("b4");
    vector<bool> vals;
    for (uint32_t i = 0; i < 4; i++) {
        vals.push_back(true);
    }

    cout << Parser::parseInput("input.sl") << endl;
    /*<uint32_t> temp = Parser::getVarValues(4,16);
    for (int i = 0; i < temp.size(); i++) {
        cout << bitset<16>(temp[i]) << endl;
    }*/
    //truthTable("(not (and (not b1) ) b2 )", vars, vals);
}

