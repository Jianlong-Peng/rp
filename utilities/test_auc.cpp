#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include "tools.h"

using namespace std;

int main(int argc, char *argv[])
{
    bool x[5] = {true,true,false,true,false};
    double y1[5] = {0.7,0.8,0.4,0.6,0.5};
    double y2[5] = {0.6,0.4,0.5,0.7,0.8};

    vector<bool> labels(x,x+5);
    vector<double> ys1(y1,y1+5);
    cout << "labels: ";
    copy(labels.begin(),labels.end(),ostream_iterator<double>(cout," "));
    cout << endl << "ys: ";
    copy(ys1.begin(),ys1.end(),ostream_iterator<double>(cout, " "));
    cout << endl << "AUC: " << calcAUC(labels, ys1) << endl;

    vector<double> ys2(y2,y2+5);
    cout << "labels: ";
    copy(labels.begin(),labels.end(),ostream_iterator<double>(cout," "));
    cout << endl << "ys: ";
    copy(ys2.begin(),ys2.end(),ostream_iterator<double>(cout, " "));
    cout << endl << "AUC: " << calcAUC(labels, ys2) << endl;

    return 0;

}
