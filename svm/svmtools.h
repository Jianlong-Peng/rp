/*=============================================================================
#     FileName: svmtools.h
#         Desc: 
#       Author: jlpeng
#        Email: jlpeng1201@gmail.com
#     HomePage: 
#      Created: 2015-03-05 20:33:20
#   LastChange: 2015-03-05 21:27:10
#      History:
=============================================================================*/
#ifndef  SVMTOOLS_H
#define  SVMTOOLS_H

#include <vector>
#include "svm.h"

svm_parameter *create_svm_parameter(int svm_type=EPSILON_SVR, int kernel_type=RBF);

// Attention:
//   1. split dataset sequentially
double svm_cv(const struct svm_problem *prob, const struct svm_parameter *para, int nfold,
        double (*eval)(const double *act, const double *pred, int n));

// Attention:
//   1. if para->svm_type is EPSILON, then combination of `c,g,p` will be searched
//      otherwise, `c,g` will be searched
//   2. after it's called, `para->C, gamma, p` will be modified
//   3. return the best result
//   4. if sign > 0, then greater `eval` returns, the better is the combination.
//      if sign < 0, then less `eval` returns, the better is the combination.
double grid_search(svm_problem *prob, svm_parameter *para, 
        int nfold, bool verbose, int sign
        double (*eval)(const double *act, const double *pred, int n));

#endif   /* ----- #ifndef SVMTOOLS_H  ----- */

