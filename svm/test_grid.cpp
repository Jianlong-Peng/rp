/*=============================================================================
#     FileName: test_grid.cpp
#         Desc: 
#       Author: jlpeng
#        Email: jlpeng1201@gmail.com
#     HomePage: 
#      Created: 2015-03-06 11:03:22
#   LastChange: 2015-03-06 12:46:37
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
            << "         each has range -8:8:1" << endl
            << endl;
        exit(EXIT_FAILURE);
    }

    int n = atoi(argv[1]);
    vector<vector<int> > range(n,vector<int>(3));

    for(int i=0; i<n; ++i) {
        range[i][0] = -8;
        range[i][1] = 8;
        range[i][2] = 1;
    }

    GridPara gp(range);
    /*
    cout << "range: ";
    copy(gp.range.begin(),gp.range.end(),ostream_iterator<int>(cout," "));
    cout << endl;

    cout << "index: ";
    copy(gp.index.begin(),gp.index.end(),ostream_iterator<int>(cout," "));
    cout << endl;
    */
    while(gp.next()) {
        copy(gp.index.begin(),gp.index.end(),ostream_iterator<int>(cout," "));
        cout << endl;
    }

    cout << endl << "another example [-6,5,1] & [6,9,1]" << endl;
    GridPara gp2(6,-6,5,1,6,9,1);
    while(gp2.next()) {
        copy(gp2.index.begin(),gp2.index.end(),ostream_iterator<int>(cout," "));
        cout << endl;
    }

    return 0;
}

