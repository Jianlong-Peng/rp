/*=============================================================================
#     FileName: svmtools.cpp
#         Desc: 
#       Author: jlpeng
#        Email: jlpeng1201@gmail.com
#     HomePage: 
#      Created: 2015-03-05 20:37:09
#   LastChange: 2015-03-06 16:38:15
#      History:
=============================================================================*/
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <string>
#include <fstream>
#include <cstdlib>
#include <cstdarg>
#include <cstring>
#include <cfloat>
#include <cmath>
#include "svm.h"
#include "svmtools.h"

using std::cout;
using std::cerr;
using std::endl;
using std::vector;
using std::copy;
using std::ostream_iterator;
using std::ifstream;
using std::string;

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

    return para;
}

svm_problem *read_svm_problem(const char *infile)
{
    ifstream inf(infile);
    if(!inf) {
        cerr << "Error: failed to open " << infile << endl;
        exit(EXIT_FAILURE);
    }
    svm_problem *prob = (svm_problem*)malloc(sizeof(svm_problem));
    prob->l = 0;
    int elements = 0;
    string line;
    while(getline(inf,line)) {
        for(string::size_type i=0; i<line.size(); ++i) {
            if(line[i]==' ' || line[i]=='\t')
                ++elements;
        }
        ++elements;
        prob->l++;
    }
    inf.close();

    prob->y = (double*)malloc(sizeof(double)*(prob->l));
    prob->x = (svm_node**)malloc(sizeof(svm_node*)*(prob->l));
    svm_node *x_space = (svm_node*)malloc(sizeof(svm_node)*elements);

    ifstream inf2(infile);
    int k=0,l=0;
    string::size_type i,j,z;
    while(getline(inf2,line)) {
        i = line.find_first_of(" \t");
        prob->y[l] = atof(line.substr(0,i).c_str());
        prob->x[l] = &x_space[k];
        ++i;
        while(true) {
            j = line.find_first_of(" \t",i);
            z = line.find_first_of(":",i);
            x_space[k].index = atoi(line.substr(i,z-i).c_str());
            if(j == string::npos) {
                x_space[k].value = atof(line.substr(z+1).c_str());
                break;
            }
            else {
                x_space[k].value = atof(line.substr(z+1,j-z-1).c_str());
                i = j+1;
            }
            ++k;
        }
        x_space[k].index = -1;
        ++k;
        ++l;
    }
    inf2.close();

    return prob;
}

void free_svm_problem(svm_problem *prob)
{
    if(prob == NULL)
        return;
    if(prob->y)
        free(prob->y);
    if(prob->x) {
        free(prob->x[0]);
        free(prob->x);
    }
    free(prob); 
}

CV::CV(const svm_problem *prob1): prob(prob1)
{
    perm = (int*)malloc(sizeof(int)*(prob->l));
    for(int i=0; i<prob->l; ++i)
        perm[i] = i;
}

CV::CV(const CV &cv): prob(cv.prob)
{
    this->perm = (int*)malloc(sizeof(int)*(this->prob->l));
    memcpy(this->perm, cv.perm, sizeof(int)*(this->prob->l));
}

CV &CV::operator=(const CV &cv)
{
    if(this == &cv)
        return *this;
    if(this->perm)
        free(this->perm);
    this->prob = cv.prob;
    this->perm = (int*)malloc(sizeof(int)*(this->prob->l));
    memcpy(this->perm, cv.perm, sizeof(int)*(this->prob->l));
    return *this;
}

double *CV::run(int nfold, const svm_parameter *para)
{
    int i,j,begin, end;
    int total = prob->l;
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
            train->y[train->l] = prob->y[perm[j]];
            train->x[train->l] = prob->x[perm[j]];
            train->l++;
        }
        for(j=end; j<total; ++j) { // train
            train->y[train->l] = prob->y[perm[j]];
            train->x[train->l] = prob->x[perm[j]];
            train->l++;
        }
        // train and predict
        model = svm_train(train,para);
        for(j=begin; j<end; ++j)
            predictY[perm[j]] = svm_predict(model,prob->x[perm[j]]);
        svm_free_and_destroy_model(&model);
    }
    free(fold_start);
    free(train->y);
    free(train->x);
    free(train);
    return predictY;
}

double CV::run(int nfold, const svm_parameter *para,
        double (*eval)(const double *act, const double *pred, int n))
{
    double *target = this->run(nfold, para);
    double val = eval(prob->y, target, prob->l);
    free(target);
    return val;
}

CV::~CV()
{
    if(perm)
        free(perm);
}

void RandCV::split()
{
    for(int i=0; i<prob->l; ++i) {
        int j = rand()%(prob->l - i);
        int temp = perm[i];
        perm[i] = perm[j];
        perm[j] = temp;
    }
}

void StratifyCV::split()
{
    cout << "to be implemented" << endl;
}


GridPara::GridPara(const vector<vector<int> > &r): range(r), index(vector<int>(r.size()))
{
    for(vector<int>::size_type i=0; i<range.size(); ++i)
        index[i] = range[i][0];
    index[0] = index[0] - range[0][2];
    /*
    cout << "range: ";
    copy((this->range).begin(),(this->range).end(),ostream_iterator<int>(cout," "));
    cout << endl;
    */
}

GridPara::GridPara(int n, ...): range(vector<vector<int> >(n/3, vector<int>(3))), index(vector<int>(n/3))
{
    if(n%3 != 0) {
        cerr << "Error: number of parameters for GridPara must be 3X, but " 
            << n << " is given!" << endl;
        exit(EXIT_FAILURE);
    }
    va_list var_arg;
    va_start(var_arg, n);
    for(int i=0, j=0; i<n; ++i, j=(j+1)%3) {
        range[i/3][j] = va_arg(var_arg, int);
        if(j == 0)
            index[i/3] = range[i/3][j];
    }
    va_end(var_arg);
    index[0] = index[0] - range[0][2];
}

GridPara::GridPara(const GridPara &gp)
{
    for(vector<vector<int> >::size_type i=0; i<gp.range.size(); ++i)
        (this->range).push_back(gp.range[i]);
    this->index = vector<int>(gp.index.begin(),gp.index.end());
}

bool GridPara::next() {
    vector<int>::size_type i;
    for(i=0; i<index.size(); ++i) {
        index[i] = index[i] + range[i][2];
        if(index[i] > range[i][1])
            index[i] = range[i][0];
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
    if(this == &gp)
        return *this;

    (this->range).clear();
    for(vector<vector<int> >::size_type i=0; i<gp.range.size(); ++i)
        (this->range).push_back(gp.range[i]);
    this->index = vector<int>(gp.index.begin(),gp.index.end());

    return *this;
}

double grid_search(CV &cv, svm_parameter *para, int nfold, bool verbose, int sign,
        double (*eval)(const double *act, const double *pred, int n))
{
    double best_eval = (sign>0)?DBL_MIN:DBL_MAX;
    double curr_eval;
    double best_c=para->C, best_g=para->gamma, best_p=para->p;
    GridPara *gp(NULL);
    if(para->svm_type == EPSILON_SVR)
        gp = new GridPara(9,-8,8,1,-8,8,1,1,5,1);
    else
        gp = new GridPara(6,-8,8,1,-8,8,1);
    while(gp->next()) {
        para->C = pow(2.0, gp->index[0]);
        para->gamma = pow(2.0, gp->index[1]);
        if(para->svm_type == EPSILON_SVR)
            para->p = 0.05 * gp->index[2];
        curr_eval = cv.run(nfold, para, eval);
        if(verbose) {
            cout << "check c=" << para->C << ", g=" << para->gamma;
            if(para->svm_type == EPSILON_SVR)
                cout << ", p=" << para->p;
            cout << " => eval=" << curr_eval << endl;
        }
        if((curr_eval - best_eval)*sign > 0) {
            best_c = para->C;
            best_g = para->gamma;
            if(para->svm_type == EPSILON_SVR)
                best_p = para->p;
            best_eval = curr_eval;
        }
    }

    para->C = best_c;
    para->gamma = best_g;
    if(para->svm_type == EPSILON_SVR)
        para->p = best_p;
    if(verbose) {
        cout << "BEST c=" << para->C << ", g=" << para->gamma;
        if(para->svm_type == EPSILON_SVR)
            cout << ", p=" << para->p;
        cout << " => eval=" << best_eval << endl;
    }
    delete gp;
    return best_eval;
}

