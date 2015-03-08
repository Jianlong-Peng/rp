/*=============================================================================
#     FileName: main.cpp
#         Desc: 
#       Author: jlpeng
#        Email: jlpeng1201@gmail.com
#     HomePage: 
#      Created: 2015-03-06 14:01:30
#   LastChange: 2015-03-06 11:22:54
#      History:
=============================================================================*/
#include <iostream>
#include <vector>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include "tools.h"
#include "../svm/svm.h"
#include "../partition/tools.h"

using namespace std;

vector<double> (*expectation)(const Sample&, vector<PredictResult>&);
void print_null(const char *s) {}

int main(int argc, char *argv[])
{
    cout << "CMD:";
    for(int i=0; i<argc; ++i)
        cout << " " << argv[i];
    cout << endl;

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
#ifdef NTHREAD
            << endl << "Attention" << endl
            << "  1. this is multi-thread version" << endl
#endif
            << endl;
        exit(EXIT_FAILURE);
    }

    string train_file("");
    string som_file("");
    string out_file("");
    int estep(1), epoch(100);
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

    svm_set_print_string_function(print_null);
    srand(time(NULL));

    Sample sample;
    sample.read_problem(train_file, som_file);
    EM em(sample);
    em.init(test_som);
    em.run(epoch, 1E-3, true, expectation);

    vector<double> &contrib = em.get_fraction();
    vector<double> &cgp = em.get_cgp();
    if(out_file.empty()) {
        copy(contrib.begin(), contrib.end(), ostream_iterator<double>(cout," "));
        copy(cgp.begin(), cgp.end(), ostream_iterator<double>(cout, " "));
        cout << endl;
    }
    else {
        ofstream outf(out_file.c_str());
        if(!outf) {
            cerr << "Warning: failed to open " << out_file << endl
                << "         atom contribution will be displayed in the screen" << endl;
            copy(contrib.begin(), contrib.end(), ostream_iterator<double>(cout," "));
            copy(cgp.begin(), cgp.end(), ostream_iterator<double>(cout, " "));
            cout << endl;
        }
        else {
            copy(contrib.begin(), contrib.end(), ostream_iterator<double>(outf," "));
            copy(cgp.begin(), cgp.end(), ostream_iterator<double>(outf, " "));
            outf << endl;
            outf.close();
        }
    }
    
    return 0;
}

