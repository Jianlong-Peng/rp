/*=============================================================================
#     FileName: svmtools.h
#         Desc: 
#       Author: jlpeng
#        Email: jlpeng1201@gmail.com
#     HomePage: 
#      Created: 2015-03-05 20:33:20
#   LastChange: 2015-03-06 16:31:40
#      History:
=============================================================================*/
#ifndef  SVMTOOLS_H
#define  SVMTOOLS_H

#include <cstdarg>
#include <vector>
#include "svm.h"

svm_parameter *create_svm_parameter(int svm_type=EPSILON_SVR, int kernel_type=RBF);
svm_problem *read_svm_problem(const char *infile);
void free_svm_problem(svm_problem *prob);


class CV
{
public:
    explicit CV(const svm_problem*);
    CV(const CV&);
    CV &operator=(const CV&);
    // user-specific method to split the training set - by modify perm
    // by defualt, split the data set sequentially
    void split() {}
    // return predicted values. do not forget to free it.
    double *run(int nfold, const svm_parameter *para);
    double run(int nfold, const svm_parameter *para,
            double (*eval)(const double *act, const double *pred, int n));
    ~CV();
protected:
    int *perm;
    const svm_problem *prob;
};

class RandCV: public CV
{
public:
    RandCV(const svm_problem *prob): CV(prob) {}
    void split();
};

class StratifyCV: public CV
{
public:
    StratifyCV(const svm_problem *prob): CV(prob) {}
    void split();
};


class GridPara
{
public:
    // vector of 3-tuple (min,max,delta)
    GridPara(const std::vector<std::vector<int> > &range);
    // min,max,delta,...
    GridPara(int n, ...);
    GridPara(const GridPara&);
    bool next();
    GridPara &operator=(const GridPara &);
public:
    std::vector<std::vector<int> > range;
    // `index` for generated values
    std::vector<int> index;
};

/*
 * Parameter
 * 1. cv: instance of class CV or its derived class
 * 2. para:
 *    if para->svm_type is EPSILON_SVR, then combination of `c,g,p` will be searched
 *    otherwise, `c,g` will be searched
 * 3. nfold: int, do {nfold}-fold cross-validation
 * 4. verbose: bool, if true, display info.
 * 5. sign:
 *    if sign > 0, then greater `eval` returns, the better is the combination.
 *    if sign < 0, then less `eval` returns, the better is the combination.
 *    if sign = 0, are you kidding
 * 6. eval: user-specific function to estimate the performance of certain combination of parameters
 */
double grid_search(CV &cv, svm_parameter *para, int nfold, bool verbose, int sign,
        double (*eval)(const double *act, const double *pred, int n));

#endif   /* ----- #ifndef SVMTOOLS_H  ----- */

