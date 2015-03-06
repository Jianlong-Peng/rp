/*=============================================================================
#     FileName: test_grid.cpp
#         Desc: 
#       Author: jlpeng
#        Email: jlpeng1201@gmail.com
#     HomePage: 
#      Created: 2015-03-06 11:03:22
#   LastChange: 2015-03-06 11:15:53
#      History:
=============================================================================*/
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <cstdlib>
#include "svmtools.h"

using namespace std;

int main(int argc, char *argv[])
{
    if(argc != 2) {
        cerr  << endl << "  Usage: " << argv[0] << " n" << endl
            << "    OBJ: to generate combination of `n` numbers" << endl
            << "         each has range [-8,8]." << endl
            << endl;
        exit(EXIT_FAILURE);
    }

    int n = atoi(argv[1]);
    vector<int> range(2*n,0);

    for(int i=0; i<n; ++i) {
        range[2*i+0] = -8;
        range[2*i+1] = 8;
    }

    GridPara gp(range);

    while(gp.next()) {
        copy(gp.index.begin(),gp.index.end(),ostream_iterator<int>(cout," "));
        cout << endl;
    }

    return 0;
}

