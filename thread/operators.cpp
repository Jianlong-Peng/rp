/*=============================================================================
#     FileName: operators.cpp
#         Desc: 
#       Author: jlpeng
#        Email: jlpeng1201@gmail.com
#     HomePage: 
#      Created: 2014-09-20 10:12:52
#   LastChange: 2015-03-30 06:13:50
#      History:
=============================================================================*/
#include <iostream>
#include <cmath>
#include <cassert>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <cstdlib>
#include <cfloat>
#include <ctime>
#include <omp.h>
#include "../utilities/tools.h"
#include "../utilities/extern_tools.h"
#include "../svm/svm.h"
#include "operators.h"
#include "extern_tools.h"
#include <ga/GA1DArrayGenome.h>
#include <ga/garandom.h>


using std::cout;
using std::cerr;
using std::endl;
using std::vector;
using std::string;
using std::ifstream;
using std::ostream;
using std::min_element;
using std::max_element;
using std::copy;
using std::ostream_iterator;

extern int repeat;
extern Sample train_set;
extern int nfolds;
extern vector<vector<svm_problem*> > probs;
extern vector<vector<int> > perm;
extern vector<int> num_xs;
extern int num_types;
extern vector<svm_parameter*> para;
extern bool do_log;


bool cv_detail = false;


// if begin <= end, then use all training samples to train the model, and
// apply the model to training set
static void do_each(int begin, int end, vector<double> &actualY, 
    vector<PredictResult> &predictY, vector<int> &sample_index, int ii)
{
    int i,j,k,idx_genome;
    for(i=0; i<num_types; ++i)
        probs[ii][i]->l = 0;

    if(para[ii]->kernel_type == PRECOMPUTED) {
        cerr << "Error: PRECOMPUTED kernel not supported!!!" << endl;
        exit(EXIT_FAILURE);
    }
    else {
        // construct train and test set
        for(i=0; i<train_set.num_samples(); ++i) {
            // test set
            if(i>=begin && i<end)
                continue;
            // training set
            idx_genome = train_set.get_start_index(perm[ii][i]);
            for(j=0; j<train_set[perm[ii][i]].num_atoms; ++j) {
                int _type = train_set[perm[ii][i]].atom_type[j];
                for(k=0; k<num_xs[_type]; ++k) {
                    probs[ii][_type]->x[probs[ii][_type]->l][k].index = k+1;
                    probs[ii][_type]->x[probs[ii][_type]->l][k].value = train_set[perm[ii][i]].x[j][k];
                }
                probs[ii][_type]->x[probs[ii][_type]->l][k].index = -1;
                if(do_log)
                    probs[ii][_type]->y[probs[ii][_type]->l] = log10(population[idx_genome]) + train_set[perm[ii][i]].y;
                else
                    probs[ii][_type]->y[probs[ii][_type]->l] = pow(10, train_set[perm[ii][i]].y)*population[idx_genome];
                probs[ii][_type]->l++;
                ++idx_genome;
            }
        }
        // train models
        idx_genome = train_set.get_start_index(-1);
        vector<svm_model*> models(num_types, NULL);
        for(i=0; i<num_types; ++i) {
            para[ii]->C = population[idx_genome];
            para[ii]->gamma = population[idx_genome+1];
            para[ii]->p = population[idx_genome+2];
            idx_genome += 3;
            models[i] = svm_train(probs[ii][i], para[ii]);
        }

        // predict
        int max_num_xs = *max_element(num_xs.begin(), num_xs.end());
        struct svm_node *x = (struct svm_node*)malloc(sizeof(struct svm_node)*(max_num_xs+1));
        if(begin >= end) {
            begin = 0;
            end = train_set.num_samples();
        }
        for(i=begin; i<end; ++i) {
            PredictResult val;
            val.y = 0.;
            //double val = 0.;
            for(j=0; j<train_set[perm[ii][i]].num_atoms; ++j) {
                int _type = train_set[perm[ii][i]].atom_type[j];
                for(k=0; k<num_xs[_type]; ++k) {
                    x[k].index = k+1;
                    x[k].value = train_set[perm[ii][i]].x[j][k];
                }
                x[k].index = -1;
                double each_value = svm_predict(models[_type], x);
                if(do_log) {
                    val.each_y.push_back(each_value);
                    val.y += pow(10, each_value);
                }
                else {
                    if(each_value < 0) {
                        cout << "Warning(" << __FILE__ << ":" << __LINE__ << "): predicted atom contribution < 0" << endl;
                        val.each_y.push_back(each_value);
                    }
                    else
                        val.each_y.push_back(log10(each_value));
                    val.y += each_value;
                }
                if(!train_set[perm[ii][i]].som.empty())
                    val.som.push_back(train_set[perm[ii][i]].som[j]);
            }
            if(val.y < 0)
                cout << "Warning(" << __FILE__ << ":" << __LINE__ << "): predicted CL < 0, may be out of range!" << endl;
            val.y = log10(val.y);
            sample_index.push_back(perm[ii][i]);
            actualY.push_back(train_set[perm[ii][i]].y);
            predictY.push_back(val);
        }

        // free
        free(x);
        for(i=0; i<num_types; ++i)
            svm_free_and_destroy_model(&models[i]);
    }
    
}


float myEvaluator(GAGenome &genome)
{
    GA1DArrayGenome<float> &g = DYN_CAST(GA1DArrayGenome<float>&, genome);
    
    myIndex = new CandidateIndices(repeat);
    obj_values.clear();
    obj_values.resize(repeat);
    population.clear();
    for(int i=0; i<train_set.count_total_num_atoms()+num_types*3; ++i)
        population.push_back(g.gene(i));

    #pragma omp parallel for schedule(static)
    for(int i=0; i<repeat; ++i) {
        int n = train_set.num_samples();
        vector<double> actualY;
        vector<PredictResult> predictY;
        vector<int> sample_index;
        for(int j=0; j<nfolds; ++j) {
            int begin = j*n/nfolds;
            int end   = (j+1)*n/nfolds;
            do_each(begin, end, actualY, predictY, sample_index, i);
        }
        obj_values[i] = obj(actualY, predictY, sample_index, population, false);
        if(cv_detail) {
            double r = calcR(actualY,predictY);
            double rmse = calcRMSE(actualY,predictY);
            #pragma omp critical
            cout << " (rmse=" << rmse << " r=" << r << ")";
        }
    }
    
#ifdef TEST_OUTPUT
    cout << "obj_values: ";
    copy(obj_values.begin(), obj_values.end(), ostream_iterator<float>(cout, " "));
    cout << endl;
#endif

    return (accumulate(obj_values.begin(), obj_values.end(), 0.) / repeat);
}

