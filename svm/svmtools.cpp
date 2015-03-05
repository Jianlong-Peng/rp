/*=============================================================================
#     FileName: svmtools.cpp
#         Desc: 
#       Author: jlpeng
#        Email: jlpeng1201@gmail.com
#     HomePage: 
#      Created: 2015-03-05 20:37:09
#   LastChange: 2015-03-05 21:26:11
#      History:
=============================================================================*/
#include <iostream>
#include <vector>
#include <cstdlib>
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

class GridPara
{
public:
    GridPara(): base_c(2.0),base_g(2.0),base_p(2.0) {}
    bool next() {
        if(++(self.ci) > 8) {
            self.ci = -8;
            if(base_gi == 0.)
                return false;
            if(++(self.gi) > 8) {
                if(base_pi == 0.)
                    return false;
                self.gi = -8;
                if(++(self.pi) > 5)
                    return false;
            }
        }
        return true;
    }

public:
    double base_c;
    double base_g;
    double base_p;
    int ci;
    int gi;
    int pi;
}

#define BASE_C 2.0
#define BASE_G 2.0
#define BASE_P 0.05

double grid_search(svm_problem *prob, svm_parameter *para, 
        int nfold, bool verbose, int sign,
        double (*eval)(const double *act, const double *pred, int n))
{
    double best_eval = (sign>0)?DBL_MIN:DBL_MAX;
    double curr_eval;
    double best_c=para->C, best_g=para->gamma, best_p=para->p;
    //double *target = (double*)malloc(sizeof(double)*(prob->l));
    GridPara gp;
    if(para->svm_type != EPSILON_SVR)
        gp.base_p = 0.;
    while(gp.next()) {
        para->C = pow(gp.base_c, gp.ci);
        para->gamma = pow(gp.base_g, gp.gi);
        para->p = gp.base_p * gp.pi;
        //svm_cross_validation(prob,para,nfold,target);
        //curr_eval = eval(prob->y,target,prob->l);
        curr_eval = svm_CV(para,prob,nfold,eval);
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
    return best_eval;
}

