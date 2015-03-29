/*=============================================================================
#     FileName: extern_tools.cpp
#         Desc:
#       Author: jlpeng
#        Email: jlpeng1201@gmail.com
#     HomePage:
#      Created: 2015-03-18 10:52:13
#   LastChange: 2015-03-18 10:53:21
#      History:
=============================================================================*/

#include "tools.h"
#include <vector>
#include <numeric>
#include <iostream>
#include <cmath>

using std::vector;
using std::accumulate;
using std::cout;
using std::endl;

extern bool calc_auc;
extern bool calc_iap;
extern bool calc_consistency;
extern bool calc_x2;
extern Sample train_set;
extern double belta;
extern double wx2;
extern double wauc;

float obj(vector<double> &actualY, vector<PredictResult> &predictY,
        vector<int> &sample_index, vector<float> &population, bool verbose)
{
    int n = static_cast<int>(actualY.size());
    double mrss = calcRSS(actualY, predictY) / n;
    double mauc=0., miap=0., mdelta=0., mean_x2=0.;
    if(calc_auc) {
        int k = 0;
        for(vector<PredictResult>::size_type i=0; i<predictY.size(); ++i) {
            if(!predictY[i].som.empty()) {
                mauc += calcAUC(predictY[i].som, predictY[i].each_y);
                ++k;
            }
        }
        mauc /= k;
    }
    if(calc_iap) {
        vector<double> iap = calcIAP(actualY, predictY);
        miap = accumulate(iap.begin(), iap.end(), 0.) / iap.size();
    }
    if(calc_consistency) {
        int k = 0;
        for(vector<PredictResult>::size_type i=0; i<predictY.size(); ++i) {
            int idx_genome = train_set.get_start_index(sample_index[i]);
            for(vector<double>::size_type j=0; j<predictY[i].each_y.size(); ++j) {
                double temp_actual  = actualY[i] + log10(population[idx_genome]);
                double temp_predict = predictY[i].each_y[j];
                mdelta += pow(temp_predict-temp_actual, 2);
                ++k;
                ++idx_genome;
            }
        }
        mdelta /= k;
    }
    if(calc_x2) {
        for(vector<PredictResult>::size_type i=0; i<predictY.size(); ++i) {
            double val=0.;
            int idx_genome = train_set.get_start_index(sample_index[i]);
            for(vector<double>::size_type j=0; j<predictY[i].each_y.size(); ++j) {
                val += pow(population[idx_genome],2);
                ++idx_genome;
            }
            mean_x2 += val;
        }
        mean_x2 /= predictY.size();
    }
    if(verbose) {
        cout << "mrss=" << mrss << " mauc=" << mauc << " miap=" << miap << " mdelta=" << mdelta << " mean_x2=" << mean_x2 << " OBJ=" << (1./mrss+mauc+miap+belta/mdelta+wx2*mean_x2) << endl;
    }
    if(mrss < 1e-3)
        mrss = 1e-3;
    if(mdelta < 1e-3)
        mdelta = 1e-3;
    return static_cast<float>(1./mrss+wauc*mauc+miap+belta/mdelta+wx2*mean_x2);
}


