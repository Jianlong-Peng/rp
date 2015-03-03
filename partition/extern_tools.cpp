/*=============================================================================
#     FileName: extern_tools.cpp
#         Desc: 
#       Author: jlpeng
#        Email: jlpeng1201@gmail.com
#     HomePage: 
#      Created: 2014-10-27 13:28:22
#   LastChange: 2014-10-27 14:36:29
#      History:
=============================================================================*/
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <cmath>
#include "extern_tools.h"
#include "tools.h"
#include "svm.h"
#include <ga/garandom.h>

using namespace std;

extern int num_types;
extern Sample train_set;
extern vector<svm_problem*> probs;
extern svm_parameter *para;
extern vector<int> perm;
extern vector<int> num_each_sample;
//extern vector<vector<int> > sample_atom_index;  // index of each atom according to its type
extern vector<int> num_xs;
extern int kernel_type;
extern vector<vector<vector<double> > > kernel_matrix;

void randomize_samples(bool verbose)
{
    int n = train_set.num_samples();
    perm.clear();
    perm.resize(n);
    for(int i=0; i<n; ++i)
        perm[i] = i;
    for(int i=0; i<n; ++i) {
        int j = GARandomInt(i, n-1);
        swap(perm[i],perm[j]);
    }
    if(verbose) {
        cout << "perm: ";
        copy(perm.begin(), perm.end(), ostream_iterator<int>(cout," "));
        cout << endl;
    }
}

static double calcKernel(vector<double> &x, vector<double> &y)
{
    if(kernel_type == 1)
        return tanimotoKernel(x,y);
    else if(kernel_type == 2)
        return minMaxKernel(x,y);
    else {
        cerr << "Error: in calcKernel, only kernel_type 1 or 2 allowed, but it's " << kernel_type << endl;
        exit(EXIT_FAILURE);
    }
}

void construct_svm_problems_parameters()
{
    // num_types & num_xs & number of atoms of type i
    num_types = 0;
    for(int i=0; i<train_set.num_samples(); ++i)
        for(int j=0; j<train_set[i].num_atoms; ++j)
            num_types = (train_set[i].atom_type[j]>num_types)?(train_set[i].atom_type[j]):num_types;
    num_types += 1;
    num_xs.resize(num_types, 0);
    num_each_sample.resize(num_types, 0);
    for(int i=0; i<train_set.num_samples(); ++i) {
        //vector<int> sample_index_temp;
        for(int j=0; j<train_set[i].num_atoms; ++j) {
            int _type = train_set[i].atom_type[j];
            int tmp = static_cast<int>(train_set[i].x[j].size());
            if(num_xs[_type] && tmp!=num_xs[_type]) {
                cerr << "Error: incompatible number of Xs for atom type " << _type << endl;
                exit(EXIT_FAILURE);
            }
            num_xs[_type] = tmp;
            //sample_index_temp.push_back(num_each_sample[train_set[i].atom_type[j]]);
            num_each_sample[train_set[i].atom_type[j]] += 1;
        }
        //sample_atom_index.push_back(sample_index_temp);
    }

    // svm problems
    probs.resize(num_types);
    for(int i=0; i<num_types; ++i) {
        cout << "for atom type " << i << ", there are " << num_each_sample[i] << " samples and "
            << num_xs[i] << " Xs" << endl;
        probs[i] = (svm_problem*)malloc(sizeof(svm_problem));
        probs[i]->l = 0;
        probs[i]->y = (double*)malloc(sizeof(double)*(num_each_sample[i]));
        probs[i]->x = (struct svm_node**)malloc(sizeof(struct svm_node*)*(num_each_sample[i]));
        for(int j=0; j<num_each_sample[i]; ++j) {
            if(kernel_type != 0)
                probs[i]->x[j] = (struct svm_node*)malloc(sizeof(struct svm_node)*(num_each_sample[i]+2));
            else
                probs[i]->x[j] = (struct svm_node*)malloc(sizeof(struct svm_node)*(num_xs[i]+1));
        }
    }

    // pre-calculate kernel matrix
    if(kernel_type != 0) {
        kernel_matrix.clear();
        for(int i=0; i<num_types; ++i) {
            vector<vector<double> > tempX;
            for(int j=0; j<train_set.num_samples(); ++j) {
                for(int k=0; k<train_set[j].num_atoms; ++k) {
                    if(train_set[j].atom_type[k] == i)
                        tempX.push_back(train_set[j].x[k]);
                }
            }
            vector<vector<double> > matrix;
            for(vector<vector<double> >::size_type j=0; j<tempX.size(); ++j) {
                vector<double> temp(num_each_sample[i]);
                for(vector<vector<double> >::size_type k=0; k<tempX.size(); ++k)
                    temp[k] = calcKernel(tempX[j], tempX[k]);
                matrix.push_back(temp);
            }
            kernel_matrix.push_back(matrix);
        }
    }

    if(para != NULL)
        free(para);
    para = (struct svm_parameter*)malloc(sizeof(struct svm_parameter));
    para->svm_type = EPSILON_SVR;
    if(kernel_type != 0)
        para->kernel_type = PRECOMPUTED;
    else
        para->kernel_type = RBF;
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

void free_svm_problems_parameters()
{
    for(int i=0; i<num_types; ++i) {
        free(probs[i]->y);
        for(int j=0; j<num_each_sample[i]; ++j)
            free(probs[i]->x[j]);
        free(probs[i]->x);
        free(probs[i]);
    }
    probs.clear();

    svm_destroy_param(para);
    para = NULL;
}

// if begin >= end, then train models using all samples, followed by predicting the whole training set
// otherwise, train_set[begin:end] will be used as test set, and the remaining being training set
void do_each(int begin, int end, vector<double> &actualY, vector<PredictResult> &predictY,
        vector<float> &population, bool fraction)
{
    int i,j,k,idx_genome;
    for(i=0; i<num_types; ++i)
        probs[i]->l = 0;

    if(para->kernel_type == PRECOMPUTED) {
        vector<vector<vector<double> > > train_xs(num_types);
        vector<vector<double> > train_ys(num_types);
        // construct training set
        idx_genome = 0;
        for(i=0; i<train_set.num_samples(); ++i) {
            for(j=0; j<train_set[perm[i]].num_atoms; ++j) {
                int _type = train_set[perm[i]].atom_type[j];
                if(i<begin || i>=end) {
                    train_xs[_type].push_back(train_set[perm[i]].x[j]);
                    if(fraction)
                        train_ys[_type].push_back(log10(population[idx_genome]) + train_set[perm[i]].y);
                    else
                        train_ys[_type].push_back(population[idx_genome]);
                }
                ++idx_genome;
            }
        }
        for(i=0; i<num_types; ++i) {
            int num_train = static_cast<int>(train_xs[i].size());
            for(j=0; j<num_train; ++j) {
                probs[i]->x[probs[i]->l][0].index = 0;
                probs[i]->x[probs[i]->l][0].value = j+1;
                for(k=0; k<num_train; ++k) {
                    probs[i]->x[probs[i]->l][k+1].index = k+1;
                    probs[i]->x[probs[i]->l][k+1].value = calcKernel(train_xs[i][j], train_xs[i][k]);
                }
                probs[i]->x[probs[i]->l][k+1].index = -1;
                probs[i]->y[probs[i]->l] = train_ys[i][j];
                probs[i]->l++;
            }
        }
        // 2. train svm model
        vector<svm_model*> models(num_types, NULL);
        for(i=0; i<num_types; ++i) {
            para->C = population[idx_genome];
            para->gamma = population[idx_genome+1];
            para->p = population[idx_genome+2];
            idx_genome += 3;
            models[i] = svm_train(probs[i], para);
        }
        // 3. predict
        int max_xs = *max_element(num_each_sample.begin(), num_each_sample.end());
        struct svm_node *x = (struct svm_node*)malloc(sizeof(struct svm_node)*(max_xs+2));
        if(begin >= end) {
            begin = 0;
            end = train_set.num_samples();
        }
        for(i=0; i<train_set.num_samples(); ++i) {
            if(i<begin || i>=end)
                continue;
            PredictResult val;
            val.y = 0.;
            //double val = 0.;
            for(j=0; j<train_set[perm[i]].num_atoms; ++j) {
                int _type = train_set[i].atom_type[j];
                x[0].index = 0;
                for(k=0; k<num_each_sample[_type]; ++k) {
                    x[k+1].index = k+1;
                    x[k+1].value = calcKernel(train_set[i].x[j], train_xs[_type][k]);
                }
                x[k+1].index = -1;
                double each_value = svm_predict(models[_type], x);
                val.each_y.push_back(each_value);
                val.y += pow(10, each_value);
                //val += pow(10, each_value);
            }
            if(val.y < 0)
                cout << "Warning(" << __FILE__ << ":" << __LINE__ << "): sum(10^eachy) < 0, may be out of range!" << endl;
            val.y = log10(val.y);
            actualY.push_back(train_set[perm[i]].y);
            predictY.push_back(val);
        }
        free(x);
        for(i=0; i<num_types; ++i)
            svm_free_and_destroy_model(&models[i]);
    }
    else {
        // construct train and test set
        idx_genome = 0;
        for(i=0; i<train_set.num_samples(); ++i) {
            // test set
            if(i>=begin && i<end) {
                idx_genome += train_set[perm[i]].num_atoms;
                continue;
            }
            // training set
            for(j=0; j<train_set[perm[i]].num_atoms; ++j) {
                int _type = train_set[perm[i]].atom_type[j];
                for(k=0; k<num_xs[_type]; ++k) {
                    probs[_type]->x[probs[_type]->l][k].index = k+1;
                    probs[_type]->x[probs[_type]->l][k].value = train_set[perm[i]].x[j][k];
                }
                probs[_type]->x[probs[_type]->l][k].index = -1;
                if(fraction)
                    probs[_type]->y[probs[_type]->l] = log10(population[idx_genome]) + train_set[perm[i]].y;
                else
                    probs[_type]->y[probs[_type]->l] = population[idx_genome];
                probs[_type]->l++;
                ++idx_genome;
            }
        }
        // train models
        vector<svm_model*> models(num_types, NULL);
        for(i=0; i<num_types; ++i) {
            para->C = population[idx_genome];
            para->gamma = population[idx_genome+1];
            para->p = population[idx_genome+2];
            idx_genome += 3;
            models[i] = svm_train(probs[i], para);
        }

        // predict
        /*
        for(i=begin; i<end; ++i) {
            actualY.push_back(train_set[perm[i]].y);
            predictY.push_back(train_set[perm[i]].predict(models));
        }
        */
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
            for(j=0; j<train_set[perm[i]].num_atoms; ++j) {
                int _type = train_set[perm[i]].atom_type[j];
                for(k=0; k<num_xs[_type]; ++k) {
                    x[k].index = k+1;
                    x[k].value = train_set[perm[i]].x[j][k];
                }
                x[k].index = -1;
                double each_value = svm_predict(models[_type], x);
                val.each_y.push_back(each_value);
                val.y += pow(10, each_value);
                //val += pow(10, each_value);
                if(!train_set[perm[i]].som.empty())
                    val.som.push_back(train_set[perm[i]].som[j]);
            }
            if(val.y < 0)
                cout << "Warning(" << __FILE__ << ":" << __LINE__ << "): sum(10^eachy) < 0, may be out of range!" << endl;
            val.y = log10(val.y);
            actualY.push_back(train_set[perm[i]].y);
            predictY.push_back(val);
        }

        // free
        free(x);
        for(i=0; i<num_types; ++i)
            svm_free_and_destroy_model(&models[i]);
    }
    
}

void doCV(int nfolds, vector<double> &actualY, vector<PredictResult> &predictY,
        vector<float> &population, bool fraction)
{
    int i;
    int n = train_set.num_samples();
    
    actualY.clear();
    predictY.clear();
    for(i=0; i<nfolds; ++i) {
        int begin = i*n/nfolds;
        int end   = (i+1)*n/nfolds;
        do_each(begin, end, actualY, predictY, population, fraction);
    }
}


