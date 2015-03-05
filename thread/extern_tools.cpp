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
#include "../svm/svm.h"
#include <ga/garandom.h>

using namespace std;

extern int repeat;
extern int num_types;
extern Sample train_set;
extern vector<vector<svm_problem*> > probs;
extern vector<svm_parameter *> para;
extern vector<vector<int> > perm;
extern vector<int> num_each_sample;
//extern vector<vector<int> > sample_atom_index;  // index of each atom according to its type
extern vector<int> num_xs;
extern int kernel_type;
extern vector<vector<vector<double> > > kernel_matrix;


// do it P times
void randomize_samples(bool verbose)
{
    int n = train_set.num_samples();
    perm.clear();
    perm.resize(repeat);
    cout << "to pre-randomize samples " << repeat << " times" << endl;
	GARandomSeed();
	cout << "GAGetRandomSeed()=" << GAGetRandomSeed() << endl;
    for(int ii=0; ii<repeat; ++ii) {
        perm[ii].clear();
        perm[ii].resize(n);
        for(int j=0; j<n; ++j)
            perm[ii][j] = j;
        for(int j=0; j<n; ++j) {
            int k = GARandomInt(j, n-1);
            swap(perm[ii][j],perm[ii][k]);
        }
        if(verbose) {
            cout << "perm[" << ii << "]: ";
            copy(perm[ii].begin(), perm[ii].end(), ostream_iterator<int>(cout," "));
            cout << endl;
        }
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

    // create svm problems P times
    // svm problems
    probs.clear();
    probs.resize(repeat);
    cout << "to pre-create " << repeat << " svm problems" << endl;
	for(int i=0; i<num_types; ++i) {
	    cout << "for atom type " << i << ", there are " << num_each_sample[i] << " samples and "
			<< num_xs[i] << " Xs" << endl;
	}
    for(int ii=0; ii<repeat; ++ii) {
        probs[ii].resize(num_types);
        for(int i=0; i<num_types; ++i) {
            probs[ii][i] = (svm_problem*)malloc(sizeof(svm_problem));
            probs[ii][i]->l = 0;
            probs[ii][i]->y = (double*)malloc(sizeof(double)*(num_each_sample[i]));
            probs[ii][i]->x = (struct svm_node**)malloc(sizeof(struct svm_node*)*(num_each_sample[i]));
            for(int j=0; j<num_each_sample[i]; ++j) {
                if(kernel_type != 0)
                    probs[ii][i]->x[j] = (struct svm_node*)malloc(sizeof(struct svm_node)*(num_each_sample[i]+2));
                else
                    probs[ii][i]->x[j] = (struct svm_node*)malloc(sizeof(struct svm_node)*(num_xs[i]+1));
            }
        }
    }

    // pre-calculate kernel matrix  --  not supported!!!!!!
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

    // craete para P times
    if(!para.empty()) {
        for(vector<svm_parameter*>::size_type i=0; i<para.size(); ++i)
            free(para[i]);
        para.clear();
    }
    para.resize(repeat);
    for(int ii=0; ii<repeat; ++ii) {
        para[ii] = (struct svm_parameter*)malloc(sizeof(struct svm_parameter));
        para[ii]->svm_type = EPSILON_SVR;
        if(kernel_type != 0)
            para[ii]->kernel_type = PRECOMPUTED;
        else
            para[ii]->kernel_type = RBF;
        para[ii]->degree = 3;
        para[ii]->gamma = 0.;
        para[ii]->coef0 = 0;
        para[ii]->cache_size = 100;
        para[ii]->eps = 0.001;
        para[ii]->C = 1.0;
        para[ii]->nr_weight = 0;
        para[ii]->weight_label = NULL;
        para[ii]->weight = NULL;
        para[ii]->nu = 0.5;
        para[ii]->p = 0.1;
        para[ii]->shrinking = 1;
        para[ii]->probability = 0;
    }

}

// 
void free_svm_problems_parameters()
{
    for(int ii=0; ii<repeat; ++ii) {
        for(int i=0; i<num_types; ++i) {
            free(probs[ii][i]->y);
            for(int j=0; j<num_each_sample[i]; ++j)
                free(probs[ii][i]->x[j]);
            free(probs[ii][i]->x);
            free(probs[ii][i]);
        }
        probs[ii].clear();

        svm_destroy_param(para[ii]);
        para[ii] = NULL;
    }
    probs.clear();
    para.clear();
}

