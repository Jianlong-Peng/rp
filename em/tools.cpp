/*=============================================================================
#     FileName: tools.cpp
#         Desc: 
#       Author: jlpeng
#        Email: jlpeng1201@gmail.com
#     HomePage: 
#      Created: 2015-03-05 15:29:44
#   LastChange: 2015-03-05 21:29:59
#      History:
=============================================================================*/
#include <iostream>
#include <cstdlib>
#include <vector>
#include "tools.h"
#include "../partition/tools.h"
#include "../svm/svm.h"
#include "../svm/svmtools.h"

using namespace std;

EM::EM(Sample &sample): _sample(sample)
{
    int num_types = 0;
    for(int i=0; i<_sample.num_samples(); ++i)
        for(int j=0; j<_sample[i].num_atoms; ++j)
            num_types = (_sample[i].atom_type[j]>num_types)?(_sample[i].atom_type[j]):num_types;
    num_types += 1;
    vector<int> num_xs(num_types, 0);
    vector<int> num_each_sample.resize(num_types, 0);
    for(int i=0; i<_sample.num_samples(); ++i) {
        for(int j=0; j<_sample[i].num_atoms; ++j) {
            int _type = _sample[i].atom_type[j];
            int tmp = static_cast<int>(_sample[i].x[j].size());
            if(num_xs[_type] && tmp!=num_xs[_type]) {
                cerr << "Error: incompatible number of Xs for atom type " << _type << endl;
                exit(EXIT_FAILURE);
            }
            num_xs[_type] = tmp;
            num_each_sample[_sample[i].atom_type[j]] += 1;
        }
    }

    // init svm problems
    _probs.resize(num_types);
    for(int i=0; i<num_types; ++i) {
        cout << "for atom type " << i << ", there are " << num_each_sample[i] << " samples and "
            << num_xs[i] << " Xs" << endl;
        _probs[i] = (svm_problem*)malloc(sizeof(svm_problem));
        _probs[i]->l = 0;
        _probs[i]->y = (double*)malloc(sizeof(double)*(num_each_sample[i]));
        _probs[i]->x = (struct svm_node**)malloc(sizeof(struct svm_node*)*(num_each_sample[i]));
        for(int j=0; j<num_each_sample[i]; ++i)
            _probs[i]->x[j] = (struct svm_node*)malloc(sizeof(struct svm_node)*(num_xs[i]+1));
    }

    // fill svm problems
    for(int i=0; i<_sample.num_samples(); ++i) {
        for(int j=0; j<_sample[i].num_atoms; ++j) {
            int _type = _sample[i].atom_type[j];
            int k;
            for(k=0; k<num_xs[_type]; ++k) {
                _probs[_type]->x[_probs[_type]->l][k].index = k+1;
                _probs[_type]->x[_probs[_type]->l][k].value = _sample[i].x[j][k];
            }
            _probs[_type]->x[_probs[_type]->l][k].index = -1;
            _probs[_type]->l++;
        }
    }
}

void EM::init(bool test_som)
{
    // fill _fraction
    _fraction.resize(_sample.count_total_num_atoms(),0.);
    int k = 0;
    double temp;
    for(int i=0; i<_sample.num_samples(); ++i) {
        double sum = 0.;
        // randomly initialization
        for(int j=0; j<_sample[i].num_atoms; ++j) {
            if(test_som) {
                if(!_sample[i].som.empty() && _sample[i].som[j])
                    temp = RANDOM(0.5,1.);
                else
                    temp = RANDOM(0.,0.5);
            }
            else
                temp = RANDOM(0.,1.);
            sum += temp;
            _fraction[k+j] = temp;
        }
        // make sure summation over all atoms of molecule i equal to be 1
        for(int j=0; j<_sample[i].num_atoms; ++j)
            _fraction[k+j] /= sum;
        k += _sample[i].num_atoms;
    }
}

void EM::fill_ys()
{
    vector<int> num_samples(_probs.size(),0);
    for(vector<svm_problem*>::size_type i=0; i<_probs.size(); ++i) {
        num_samples[i] = _probs[i]->l;
        _probs[i]->l = 0;
    }

    int k = 0;
    for(int i=0; i<_sample.num_samples(); ++i) {
        for(int j=0; j<_sample[i].num_atoms; ++j) {
            int _type = _sample[i].atom_type[j];
            _probs[_type]->y[_probs[_type]->l] = log10(_fraction[k]) + _sample[i].y;
            ++k;
            _probs[_type]->l++;
        }
    }

    for(vector<svm_problem*>::size_type i=0; i<_probs.size(); ++i) {
        if(num_samples[i] != _probs[i]->l)
            cerr << "Error(EM::fil_ys): different number of samples!" << endl;
    }
}

static double calc_delta(const vector<double> &v1, const vector<double> &v2)
{
    double val = 0.;
    for(vector<double>::size_type i=0; i<v1.size(); ++i)
        val += abs(v1[i]-v2[i]);
    val /= v1.size();
    return val;
}
static double calcRSS(const double *act, const double *pred, int n)
{
    double val = 0.;
    for(int i=0; i<n; ++i)
        val += pow(act[i]-pred[i],2);
    return val;
}

vector<svm_model*> train_models(vector<svm_problem*> &probs, svm_parameter *param)
{
    svm_parameter *para;
    vector<svm_model*> models(probs.size(),NULL);
    for(vector<svm_problem*>::size_type i=0; i<probs.size(); ++i) {
        grid_search(probs[i],param,5,false,-1,calcRSS);
        models[i] = svm_train(probs[i],param);
    }
}

vector<double> EM::run(int epochs, double epsilon
        vector<double> expectation(const Sample&, vector<PredictResult>&))
{
    vector<double> deltas;
    double delta(1E8);
    int iter(0);
    svm_parameter *param = create_svm_parameter();
    
    while(iter<epochs && delta<epsilon) {
        self.fill_ys();
        vector<svm_model*> models = train_models(_probs, param);
        vector<PredictResult> results = _sample.predict(models, true);
        vector<double> contrib = expectation(_sample, results);
        delta = calc_delta(_fraction, contrib);
        deltas.push_back(delta);
        for(vector<svm_model*>::size_type i=0; i<models.size(); ++i)
            svm_free_and_destroy_model(&models[i]);
        copy(contrib.begin(), contrib.end(), _fraction.begin());
        ++iter;
    }

    if(iter < epochs) {
        cout << "Warning: maximum number of iterations reached" << endl
            << "         delta=" << delta << endl;
    }
    else
        cout << "EM done with #iter=" << iter << " and delta=" << delta << endl;

    svm_destroy_param(param);

    return deltas;
}

EM::~EM(Sample &sample)
{
    for(vector<svm_problem*>::size_type i=0; i<_prob.size(); ++i) {
        free(_probs[i]->y);
        for(int j=0; j<_probs[i]->l; ++j)
            free(_probs[i]->x[j]);
        free(_probs[i]->x);
        free(_probs[i]);
    }
    _probs.clear();
}

