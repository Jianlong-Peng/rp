/*=============================================================================
#     FileName: gap_sites.cpp
#         Desc: 
#       Author: jlpeng
#        Email: jlpeng1201@gmail.com
#     HomePage: 
#      Created: 2014-10-31 13:59:33
#   LastChange: 2014-10-31 14:08:25
#      History:
=============================================================================*/
#include <iostream>
#include <fstream>
#include <cstdlib>
#include "tools.h"

using namespace std;

int main(int argc, char *argv[])
{
    if(argc != 4) {
        cerr << endl << "  Usage: " << argv[0] << " data.des data.som output" << endl
            << endl << "  data.des: file of descriptors" << endl
            << "  data.som: file of sites of metabolism" << endl
            << "  output  : where to save analyzed results" << endl
            << "            name #total_sites #observed_sites #type_0 ... #type_6" << endl
            << endl;
        exit(EXIT_FAILURE);
    }

    Sample dataset;
    dataset.read_problem(argv[1], argv[2]);

    ofstream outf(argv[3]);
    if(!outf) {
        cerr << "Error: failed to open " << argv[3] << endl;
        exit(EXIT_FAILURE);
    }

    for(int i=0; i<dataset.num_samples(); ++i) {
        vector<int> count_each(7,0);
        int count_observed = 0;
        for(int j=0; j<dataset[i].num_atoms; ++j) {
            int _type = dataset[i].atom_type[j];
            if(!dataset[i].som.empty() && dataset[i].som[j]) {
                count_each[_type] += 1;
                ++count_observed;
            }
        }
        outf << dataset[i].name << " " << dataset[i].num_atoms << " " << count_observed;
        for(vector<int>::iterator iter=count_each.begin(); iter!=count_each.end(); ++iter)
            outf << " " << *iter;
        outf << endl;
    }
    outf.close();

    return 0;
}

