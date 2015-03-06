/*=============================================================================
#     FileName: svmtools.cpp
#         Desc: 
#       Author: jlpeng
#        Email: jlpeng1201@gmail.com
#     HomePage: 
#      Created: 2015-03-05 20:37:09
#   LastChange: 2015-03-06 11:14:14
#      History:
=============================================================================*/
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstdarg>
#include <cstring>
#include "svm.h"

using std::cout;
using std::endl;
using std::vector;

svm_parameter *create_svm_parameter(int svm_type, int kernel_type)
{
    svm_parameter *para = (struct svm_parameter*)malloc(sizeof(struct svm_parameter));
    para->svm_type = svm_type;
    para->kernel_type = kernel_type;
    para->degree = 3;
    para->gamma = 0.;
    para->coef0 = 0;
    para->cache_size = 100;
    para->eps = 0.001;
    para->C = 1.0;
    para->nr_weight = 0;
    para->weight_label = NULL;
    para->weight = NULL;
    para->nu = 0.5;
    para->p = 0.1;
    para->shrinking = 1;
    para->probability = 0;
}

double svm_cv(const struct svm_problem *prob, const struct svm_parameter *para, int nfold,
        double (*eval)(const double *act, const double *pred, int n))
{
    int i,j,begin, end;
    int total = prob->l;
    double value;
    int *fold_start = (int*)malloc(sizeof(int)*(nfold+1));
    double *predictY = (double*)malloc(sizeof(double)*total);
    struct svm_problem *train = (struct svm_problem*)malloc(sizeof(struct svm_problem));
    struct svm_model *model;

    train->y = (double*)malloc(sizeof(double)*total);
    train->x = (struct svm_node**)malloc(sizeof(struct svm_node*)*total);

    for(i=0; i<=nfold; ++i)
        fold_start[i] = i*total/nfold;
    for(i=0; i<nfold; ++i) {
        begin = fold_start[i];
        end = fold_start[i+1];
        train->l = 0;
        for(j=0; j<begin; ++j) { // train
            train->y[train->l] = prob->y[j];
            train->x[train->l] = prob->x[j];
            train->l++;
        }
        for(j=end; j<total; ++j) { // train
            train->y[train->l] = prob->y[j];
            train->x[train->l] = prob->x[j];
            train->l++;
        }
        //printf("\ntraining set of %dth-fold:\n",i);
        //display_svm_problem(train);
        // train and predict
        model = svm_train(train,para);
        for(j=begin; j<end; ++j)
            predictY[j] = svm_predict(model,prob->x[j]);
        svm_free_and_destroy_model(&model);
    }
    value = eval(prob->y, predictY, prob->l);
    free(fold_start);
    free(predictY);
    free(train->y);
    free(train->x);
    free(train);
    return value;
}

GridPara::GridPara(const vector<int> &range)
{
    int n = static_cast<int>(range.size());
    if(n&1) {
        cerr << "Error: number of parameters for GridPara should be odd, but " << n << " is given!" << endl;
        exit(EXIT_FAILURE);
    }
    self.n = n / 2;
    self.range.resize(n);
    self.index.resize(self.n);
    for(int i=0; i<n; ++i) {
        self.range[i] = range[i];
        if(i&1 == 0)
            self.index[i] = range[i];
    }
    self.index[0]--;
}

GridPara::GridPara(int n, ...)
{
    if(n&1) {
        cerr << "Error: number of parameters for GridPara should be odd, but " << n << " is given!" << endl;
        exit(EXIT_FAILURE);
    }
    self.n = n / 2;
    self.range.resize(n);
    self.index.resize(self.n);
    va_list var_arg;
    va_start(var_arg, n);
    for(int i=0; i<n; ++i) {
        self.range[i] = va_arg(var_arg, int);
        if(i&1 == 0)
            self.index[i/2] = self.range[i];
    }
    va_end(var_arg);
    self.index[0]--;
}

GridPara::GridPara(const GridPara &gp)
{
    self.n = gp.n;
    self.range.resize(2*self.n);
    copy(gp.range.begin(),gp.range.end(),self.range.begin());
    self.index.resize(self.n);
    copy(gp.index.begin(),gp.index.end(),self.index.begin());
}

bool GridPara::next() {
    vector<int>::size_type i;
    for(i=0; i<index.size(); ++i) {
        if(++(index[i]) > range[2*i+1])
            index[i] = range[2*i];
        else
            break;
    }
    if(i < index.size())
        return true;
    else
        return false;
}

GridPara &GridPara::operator=(const GridPara &gp)
{
    if(this == &para)
        return *this;

    self.n = gp.n;
    self.range.resize(2*self.n);
    copy(gp.range.begin(),gp.range.end(),self.range.begin());
    self.index.resize(self.n);
    copy(gp.index.begin(),gp.index.end(),self.index.begin());

    return *this;
}


double grid_search(svm_problem *prob, svm_parameter *para, 
        int nfold, bool verbose, int sign,
        double (*eval)(const double *act, const double *pred, int n))
{
    double best_eval = (sign>0)?DBL_MIN:DBL_MAX;
    double curr_eval;
    double best_c=para->C, best_g=para->gamma, best_p=para->p;
    //double *target = (double*)malloc(sizeof(double)*(prob->l));
    GridPara *gp(NULL);
    if(para->svm_type == EPSILON_SVR)
        gp = new GridPara(6,-8,8,-8,8,1,5);
    else
        gp = new GridPara(4,-8,8,-8,8);
    while(gp.next()) {
        para->C = pow(2.0, gp.index[0]);
        para->gamma = pow(2.0, gp.index[1]);
        para->p = 0.05 * gp.index[2];
        //svm_cross_validation(prob,para,nfold,target);
        //curr_eval = eval(prob->y,target,prob->l);
        curr_eval = svm_cv(para,prob,nfold,eval);
        if(verbose) {
            cout << "check c=" << para->C << ", g=" << para->gamma << ", p=" << para->p 
                << " => eval=" << curr_eval << endl;
        }
        if((curr_eval - best_eval)*sign > 0) {
            best_c = para->C;
            best_g = para->gamma;
            best_p = para->p;
            best_eval = curr_eval;
        }
    }

    para->C = best_c;
    para->gamma = best_g;
    para->p = best_p;
    if(verbose) {
        cout << "check c=" << para->C << ", g=" << para->gamma << ", p=" << para->p 
            << " => eval=" << best_eval << endl;
    }
    //free(target);
    delete gp;
    return best_eval;
}

