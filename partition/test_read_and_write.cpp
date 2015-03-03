/*=============================================================================
#     FileName: test_read_and_write.cpp
#         Desc: 
#       Author: jlpeng
#        Email: jlpeng1201@gmail.com
#     HomePage: 
#      Created: 2014-10-25 14:19:10
#   LastChange: 2014-10-25 14:32:05
#      History:
=============================================================================*/
#include <iostream>
#include <cstdlib>
#include "tools.h"

using namespace std;

int main(int argc, char *argv[])
{
    if(argc != 4) {
        cerr << endl << "  Error: " << argv[0] << " input output out_svm" << endl
            << endl;
        exit(EXIT_FAILURE);
    }

    Sample test;
    test.read_problem(argv[1]);
    cout << endl << "totally read " << test.num_samples() << " samples and " << test.count_total_num_atoms() << " atoms from " << argv[1] << endl;
    for(int i=0; i<7; ++i)
        cout << "number of atoms of type " << i << ": " << test.number_atoms_of_type(i) << endl;
    test.write_problem(argv[2]);
    cout << "samples been written to " << argv[2] << endl;
    test.write_svm_problem(argv[3]);
    cout << "svm problems have been written to " << argv[3] << "_suffix" << endl;

    exit(EXIT_FAILURE);
}

