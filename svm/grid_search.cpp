/*=============================================================================
#     FileName: grid_search.cpp
#         Desc: 
#       Author: jlpeng
#        Email: jlpeng1201@gmail.com
#     HomePage: 
#      Created: 2015-03-06 14:38:23
#   LastChange: 2015-03-06 16:47:33
#      History:
=============================================================================*/
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <cstdlib>
#include <ctime>
#include "svm.h"
#include "svmtools.h"

using namespace std;

double calcRSS(const double *act, const double *pred, int n)
{
    double val = 0.;
    for(int i=0; i<n; ++i)
        val += pow(act[i]-pred[i],2);
    return val;
}

double calcBACC(const double *act, const double *pred, int n)
{
    double tp=0., fn=0., tn=0., fp=0.;
    for(int i=0; i<n; ++i) {
        if(act[i] == 1) {
            if(pred[i] == 1)
                tp += 1.;
            else
                fn += 1.;
        }
        else {
            if(pred[i] == 1)
                fp += 1.;
            else
                tn += 1.;
        }
    }
    double se = tp/(tp+fn);
    double sp = tn/(tn+fp);
    return (se+sp)/2.;
}

void print_null(const char *s) {}

int main(int argc, char *argv[])
{
    if(argc < 2) {
        cerr << endl << "OBJ" << endl
            << "  to do grid search for svm problem" << endl
            << endl << "Usage" << endl
            << "  " << argv[0] << " [options] in.svm" << endl
            << endl << "[options]" << endl
            << " -s svm_type: <default: 0>" << endl
            << "    0 - C-SVC" << endl
            << "    3 - epsilon-SVR" << endl
            << " -t kernel_type: <default: 2>" << endl
            << "    0 - linear: u'*v" << endl
            << "    1 - polynomial: (gamma*u'v + coef0)^degree" << endl
            << "    2 - RBF: exp(-gamma*|u-v|^2)" << endl
            << "    3 - sigmoid: tanh(gamma*u'*v + coef0)" << endl
            << " -d degree : set degree in kernel function (default 3)" << endl
            << " -wi weight: set the parameter C of class i to weight*C, for C-SVC (default 1)" << endl
            << " -n nfold: n-fold cross validation <default: 5>" << endl
            << " -v: if given, display grid search info." << endl
            << endl;
        exit(EXIT_FAILURE);
    }

    svm_set_print_string_function(print_null);
    srand(time(NULL));

    // C, gamma, p, degree

    int nfold(5);
    bool verbose(false);
    svm_parameter *para = create_svm_parameter(C_SVC, RBF);
    int i;
    for(i=1; i<argc; ++i) {
        if(argv[i][0] != '-')
            break;
        if(strcmp(argv[i],"-s") == 0)
            para->svm_type = atoi(argv[++i]);
        else if(strcmp(argv[i],"-t") == 0)
            para->kernel_type = atoi(argv[++i]);
        else if(argv[i][1] == 'w') {
            ++para->nr_weight;
            para->weight_label = (int*)realloc(para->weight_label, sizeof(int)*para->nr_weight);
            para->weight = (double*)realloc(para->weight,sizeof(double)*para->nr_weight);
            para->weight_label[para->nr_weight-1] = atoi(&argv[i][2]);
            para->weight[para->nr_weight-1] = atof(argv[++i]);
        }
        else if(argv[i][1] == 'd')
            para->degree = atoi(argv[++i]);
        else if(argv[i][1] == 'n')
            nfold = atoi(argv[++i]);
        else if(argv[i][1] == 'v')
            verbose = true;
        else {
            cerr << "Error: invalid option " << argv[i] << endl;
            exit(EXIT_FAILURE);
        }
    }
    if(argc-i != 1) {
        cerr << "Error: invalid number of arguments" << endl;
        exit(EXIT_FAILURE);
    }

    svm_problem *prob = read_svm_problem(argv[i]);
    RandCV cv(prob);
    cv.split();
    double val;
    if(para->svm_type == C_SVC)
        val = grid_search(cv, para, nfold, verbose, 1, calcBACC);
    else
        val = grid_search(cv, para, nfold, verbose, -1, calcRSS);
    cout << "c=" << para->C << ", g=" << para->gamma;
    if(para->svm_type == EPSILON_SVR)
        cout << ", p=" << para->p;
    cout << "  => best_val=" << val << endl;

    free_svm_problem(prob);
    svm_destroy_param(para);

    return 0;
}



