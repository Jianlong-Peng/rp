/*=============================================================================
#     FileName: main.cpp
#         Desc: 
#       Author: jlpeng
#        Email: jlpeng1201@gmail.com
#     HomePage: 
#      Created: 2015-03-06 14:01:30
#   LastChange: 2015-03-06 14:28:53
#      History:
=============================================================================*/
#include <iostream>
#include <vector>
#include <fstream>
#include <cstdlib>
#include "tools.h"
#include "../svm/svm.h"
#include "../partition/tools.h"

using namespace std;

vector<double> (*expectation)(const Sample&, vector<PredictResult>&);

int main(int argc, char *argv[])
{
    if(argc < 3) {
        cerr << endl << "OBJ" << endl
            << "  to optimize atom contribution using EM algorithm" << endl
            << endl << "Usage" << endl
            << "  " << argv[0] << " [options]" << endl
            << endl << "[options]" << endl
            << "  --train file" << endl
            << "  --som   file: <optional>" << endl
            << "    if given, info of experimental observed SOM will be considered" << endl
            << "  --out   file: <optional>" << endl
            << "    if given, to save optimized atom contribution (in percentage)" << endl
            << "  --estep  int: <default: 1>" << endl
            << "    1 - just rescale, contribution f_ij = y_ij / sum_j{y_ij}" << endl
            << "  --epoch  int: <default: 100>" << endl
            << "    to specify number of e- and m-step" << endl
            << endl;
        exit(EXIT_FAILURE);
    }

    string train_file("");
    string som_file("");
    string out_file("");
    int estep, epoch;
    bool test_som(false);
    int i;
    for(i=1; i<argc; ++i) {
        if(argv[i][0] != '-')
            break;
        if(strcmp(argv[i], "--train") == 0)
            train_file = argv[++i];
        else if(strcmp(argv[i], "--som") == 0) {
            som_file = argv[++i];
            test_som = true;
        }
        else if(strcmp(argv[i], "--out") == 0)
            out_file = argv[++i];
        else if(strcmp(argv[i], "--estep") == 0)
            estep = atoi(argv[++i]);
        else if(strcmp(argv[i], "--epoch") == 0)
            epoch = atoi(argv[++i]);
        else {
            cerr << "Error: invalid option " << argv[i] << endl;
            exit(EXIT_FAILURE);
        }
    }
    if(argc-i != 0) {
        cerr << "Error: invalid number of arguments" << endl;
        exit(EXIT_FAILURE);
    }
    if(train_file.empty()) {
        cerr << "Error: `--train` should be given!" << endl;
        exit(EXIT_FAILURE);
    }

    switch(estep) {
        case 1: expectation = expectation_rescale; break;
        default: cerr << "Error: inappropriate `--estep " << estep << "`" << endl; exit(EXIT_FAILURE);
    }

    Sample sample;
    sample.read_problem(train_file, som_file);
    EM em(sample);
    em.init(test_som);
    em.run(epoch, epsilon, true, expectation);

    ostream os(cout);
    if(!out_file.empty()) {
        ofstream outf(out_file.c_str());
        if(!outf) {
            cerr << "Warning: failed to open " << out_file << endl
                << "         atom contribution will be displayed in the screen" << endl;
            os = cout;
        }
        else
            os = outf;
    }
    vector<double> contrib = em.get_fraction();
    copy(contrib.begin(),contrib.end(),ostream_iterator<double>(os," "));
    os << endl;
    
    return 0;
}

