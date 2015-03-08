/*=============================================================================
#     FileName: operators.cpp
#         Desc: 
#       Author: jlpeng
#        Email: jlpeng1201@gmail.com
#     HomePage: 
#      Created: 2014-09-20 10:12:52
#   LastChange: 2015-03-03 15:18:32
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
#include <cstdlib>
#include <cfloat>
#include <ctime>
#include "tools.h"
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


extern Sample train_set;
extern int nfolds;
//extern vector<svm_problem*> probs;
//extern vector<int> perm;
//extern vector<int> num_xs;
//extern vector<int> num_each_sample;
extern int num_types;
//extern int kernel_type;
//extern vector<vector<vector<double> > > kernel_matrix;
//extern vector<vector<int> > sample_atom_index;
//extern svm_parameter *para;
extern int operator_type;
extern vector<bool> is_som;
//extern float (*obj_func)(vector<double>&, vector<PredictResult>&, GA1DArrayGenome<float>&);
extern bool calc_auc;
extern bool calc_iap;
extern bool calc_consistency;
extern bool calc_x2;
extern double belta;
extern bool do_log;


inline float get_random_c(int cmin=-8, int cmax=8)
{
    return STA_CAST(float, pow(2., GARandomInt(cmin, cmax)));
}

inline float get_random_g(int gmin=-8, int gmax=8)
{
    return STA_CAST(float, pow(2., GARandomInt(gmin, gmax)));
}
inline float get_random_p(int pmin=1, int pmax=5)
{
    return STA_CAST(float, 0.05*GARandomInt(pmin, pmax));
}

// initialize the elements for each molecule
// if operator_type == 1
//      for molecule i, randomly assign a value N_ij belong to [0,1], followed by
//      N_ij * 1/\sum_j{N_ij} such that summation over all atoms of molecule i is equal to 1
// otherwise
//      SOM sites are being initialized within [0.5,1], non-SOM sites being [0.,0.5]
void myInitializer(GAGenome &genome)
{
    GA1DArrayGenome<float> &g = DYN_CAST(GA1DArrayGenome<float> &, genome);
    g.resize(GAGenome::ANY_SIZE);
    int k = 0;
    float temp;
    for(int i=0; i<train_set.num_samples(); ++i) {
        float sum = 0.;
        // randomly initialization
        for(int j=0; j<train_set[i].num_atoms; ++j) {
            if(operator_type == 2) {
                if(is_som[k+j])
                    temp = GARandomFloat(0.5, 1.);
                else
                    temp =GARandomFloat(0., 0.5);
            }
            else
                temp = GARandomFloat(0., 1.);
            sum += temp;
            g.gene(k+j, temp);
        }
        // make sure summation over all atoms of molecule i is equal to be 1
        for(int j=0; j<train_set[i].num_atoms; ++j) {
            temp = g.gene(k+j);
            g.gene(k+j, temp/sum);
        }
        k += train_set[i].num_atoms;
    }
    // initialize c,g,p
    for(int i=0; i<num_types; ++i) {
        g.gene(k, get_random_c());
        g.gene(k+1, get_random_g());
        g.gene(k+2, get_random_p());
        k += 3;
    }
}

inline float generate_delta(float v1, float v2)
{
    float v[4] = {v1,v2,1-v1,1-v2};
    float minv = *min_element(v,v+4);
    return GARandomFloat(0., minv);
}
//swap mutator
// 1.
//   within each gene block of molecule i, randomly swap atom site i and j
//   such that summation over all atom sites of molecule i is always equal to 1!
// 2.
//   delta = min{vi, vj, 1-vi, 1-vj}
// 3.
//   if operator_type==1 or both i and j are SOM or non-SOM
//      gene(i)+/-delta, gene(i)+/-delta
//   otherwise
//      SOM site has more probability (see 0.7) to be +delta
int myMutator(GAGenome &genome, float pmut)
{
    GA1DArrayGenome<float> &g = DYN_CAST(GA1DArrayGenome<float>&, genome);
    if(pmut <= 0.)
        return 0;

    float p;
    float nMut = pmut * STA_CAST(float, g.length());
    if(nMut < 1.0) {
        nMut = 0;
        int k=0;
        for(int i=0; i<train_set.num_samples(); ++i) {
            for(int j=0; j<train_set[i].num_atoms; ++j) {
                if(GAFlipCoin(pmut)) {
                    int z = GARandomInt(0, train_set[i].num_atoms-1);
                    float delta = generate_delta(g.gene(k+j), g.gene(k+z));
                    if(operator_type==1 || (is_som[k+j]==is_som[k+z]))
                        p = 0.5;
                    else {
                        if(is_som[k+j])
                            p = 0.7;
                        else
                            p = 0.3;
                    }
                    if(GAFlipCoin(p)) {
                        g.gene(k+j, g.gene(k+j)+delta);
                        g.gene(k+z, g.gene(k+z)-delta);
                    } else {
                        g.gene(k+j, g.gene(k+j)-delta);
                        g.gene(k+z, g.gene(k+z)+delta);
                    }
                    //g.swap(k+j, GARandomInt(k, k+train_set[i].num_atoms-1));
                    ++nMut;
                }
            }
            k += train_set[i].num_atoms;
        }
    }
    else {
        for(int n=0; n<nMut; ++n) {
            int i = GARandomInt(0, train_set.num_samples()-1);
            int start = (i==0)?i:(train_set.count_num_atoms_until(i-1)-1);
            int end = start+train_set[i].num_atoms-1;
            int j = GARandomInt(start, end);
            int k = GARandomInt(start, end);
            float delta = generate_delta(g.gene(j), g.gene(k));
            if(operator_type==1 || (is_som[j]==is_som[k]))
                p = 0.5;
            else {
                if(is_som[j])
                    p = 0.7;
                else
                    p = 0.3;
            }
            if(GAFlipCoin(p)) {
                g.gene(j, g.gene(j)+delta);
                g.gene(k, g.gene(k)-delta);
            } else {
                g.gene(j, g.gene(j)-delta);
                g.gene(k, g.gene(k)+delta);
            }
            //g.swap(GARandomInt(start, end), GARandomInt(start, end));
        }
    }
    // mutation for c,g,p
    int k = train_set.count_total_num_atoms();
    for(int i=0; i<num_types; ++i) {
        if(GAFlipCoin(0.5))
            g.gene(k, get_random_c());
        if(GAFlipCoin(0.5))
            g.gene(k+1, get_random_g());
        if(GAFlipCoin(0.5))
            g.gene(k+2, get_random_p());
        k += 3;
    }

    return (STA_CAST(int,nMut));
}

// uniform crossover
// gene block of molecule i is either from mom or dad
int myCrossover(const GAGenome &p1, const GAGenome &p2, GAGenome *c1, GAGenome *c2)
{
    const GA1DArrayGenome<float> &mom = DYN_CAST(const GA1DArrayGenome<float>&, p1);
    const GA1DArrayGenome<float> &dad = DYN_CAST(const GA1DArrayGenome<float>&, p2);

    int n = 0;
    if(c1 && c2) {
        GA1DArrayGenome<float> &sis = DYN_CAST(GA1DArrayGenome<float>&, *c1);
        GA1DArrayGenome<float> &bro = DYN_CAST(GA1DArrayGenome<float>&, *c2);
        if(sis.length()!=bro.length() || 
                mom.length()!=dad.length() ||
                sis.length()!=mom.length()) {
            cerr << "Error: different gene length" << endl;
            exit(EXIT_FAILURE);
        }
        int k = 0;
        for(int i=0; i<train_set.num_samples(); ++i) {
            if(GAFlipCoin(0.5)) {
                for(int j=0; j<train_set[i].num_atoms; ++j) {
                    sis.gene(k+j, mom.gene(k+j));
                    bro.gene(k+j, dad.gene(k+j));
                }
            }
            else {
                for(int j=0; j<train_set[i].num_atoms; ++j) {
                    sis.gene(k+j, dad.gene(k+j));
                    bro.gene(k+j, mom.gene(k+j));
                }
            }
            k += train_set[i].num_atoms;
        }
        n = 2;

        // cross-over for c,g,p
        for(int i=0; i<num_types*3; ++i) {
            if(GAFlipCoin(0.5)) {
                sis.gene(k+i, mom.gene(k+i));
                bro.gene(k+i, dad.gene(k+i));
            }
            else {
                sis.gene(k+i, dad.gene(k+i));
                bro.gene(k+i, mom.gene(k+i));
            }
        }
    }
    else if(c1 || c2) {
        GA1DArrayGenome<float> &sis = (c1 ? 
                DYN_CAST(GA1DArrayGenome<float>&, *c1):
                DYN_CAST(GA1DArrayGenome<float>&, *c2));
        if(sis.length()!=mom.length() ||
                mom.length()!=dad.length()) {
            cerr << "Error: different gene length" << endl;
            exit(EXIT_FAILURE);
        }
        int k = 0;
        for(int i=0; i<train_set.num_samples(); ++i) {
            if(GAFlipCoin(0.5)) {
                for(int j=0; j<train_set[i].num_atoms; ++j)
                    sis.gene(k+j, mom.gene(k+j));
            }
            else {
                for(int j=0; j<train_set[i].num_atoms; ++j)
                    sis.gene(k+j, dad.gene(k+j));
            }
            k += train_set[i].num_atoms;
        }
        n = 1;

        // crossover for c,g,p
        for(int i=0; i<num_types*3; ++i) {
            if(GAFlipCoin(0.5))
                sis.gene(k+i, mom.gene(k+i));
            else
                sis.gene(k+i, dad.gene(k+i));
        }
    }

    return n;
}

/*
// y-value of each atom: each_y = g.gene(k) * pow(10, train_set[i].y)
// svm models are based on log10(each_y)
// predictY: log10(\sum{10^predict_each_y})
static void do_each(int begin, int end, vector<double> &actualY, vector<PredictResult> &predictY,
        GA1DArrayGenome<float> &g)
{
    int i,j,k;
    for(i=0; i<num_types; ++i)
        probs[i]->l = 0;

    // for pre-computed kernel
    if(kernel_type != 0) {
        vector<vector<int> > train_index(num_types);  // [i][]: index of type i
        vector<vector<double> > train_ys(num_types);
        vector<int> index(num_types, -1);
        int z=0;
        // construct training set
        for(i=0; i<train_set.num_samples(); ++i) {
            for(j=0; j<train_set[perm[i]].num_atoms; ++j) {
                int _type = train_set[perm[i]].atom_type[j];
                index[_type] += 1;
                if(i<begin || i>=end) {
                    train_index[_type].push_back(sample_atom_index[perm[i]][j]);
                    train_ys[_type].push_back(log10(g.gene(z)) + train_set[perm[i]].y);
                }
                ++z;
            }
        }
        for(i=0; i<num_types; ++i) {
            int num_train = STA_CAST(int, train_index[i].size());
            for(j=0; j<num_train; ++j) {
                probs[i]->x[probs[i]->l][0].index = 0;
                probs[i]->x[probs[i]->l][0].value = j+1;
                for(k=0; k<num_train; ++k) {
                    probs[i]->x[probs[i]->l][k+1].index = k+1;
                    probs[i]->x[probs[i]->l][k+1].value = kernel_matrix[i][train_index[i][j]][train_index[i][k]];
                }
                probs[i]->x[probs[i]->l][k+1].index = -1;
                probs[i]->y[probs[i]->l] = train_ys[i][j];
                probs[i]->l++;
            }
        }

        // 2. train svm models
        vector<svm_model*> models(num_types);
        for(i=0; i<num_types; ++i) {
            para->C = g.gene(z);
            para->gamma = g.gene(z+1);
            para->p = g.gene(z+2);
            z += 3;
            models[i] = svm_train(probs[i], para);
        }

        // 3. predict
        index.clear();
        index.resize(num_types, -1);
        int max_xs = *max_element(num_each_sample.begin(), num_each_sample.end());
        struct svm_node *x = (struct svm_node*)malloc(sizeof(struct svm_node)*(max_xs+2));
        for(i=0; i<train_set.num_samples(); ++i) {
            PredictResult val;
            val.y = 0.;
            for(j=0; j<train_set[perm[i]].num_atoms; ++j) {
                int _type = train_set[i].atom_type[j];
                index[_type] += 1;
                if(i>=begin && i<end) {
                    x[0].index = 0;
                    for(k=0; k<num_each_sample[_type]; ++k) {
                        x[k+1].index = k+1;
                        x[k+1].value = kernel_matrix[_type][index[_type]][train_index[_type][k]];
                    }
                    x[k+1].index = -1;
                    double each_value = svm_predict(models[_type], x);
                    val.each_y.push_back(each_value);
                    val.y += pow(10, each_value);
                    if(!train_set[perm[i]].som.empty())
                        val.som.push_back(train_set[perm[i]].som[j]);
                }
            }
            if(i>=begin && i<end) {
                val.y = log10(val.y);
                actualY.push_back(train_set[perm[i]].y);
                predictY.push_back(val);
            }
        }

        // free
        free(x);
        for(i=0; i<num_types; ++i)
            svm_free_and_destroy_model(&models[i]);
    }
    else {
        int z = 0;
        // 1. get samples and c,g,p
        for(i=0; i<train_set.num_samples(); ++i) {
            // test set
            if(i>=begin && i<end) {
                z += train_set[perm[i]].num_atoms;
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
                probs[_type]->y[probs[_type]->l] = log10(g.gene(z)) + train_set[perm[i]].y;
                probs[_type]->l++;
                z++;
            }
        }
    
        // 2. train svm model
        vector<svm_model*> models(num_types);
        for(i=0; i<num_types; ++i) {
            para->C = g.gene(z);
            para->gamma = g.gene(z+1);
            para->p = g.gene(z+2);
            z += 3;
            models[i] = svm_train(probs[i], para);
        }
    
        // 3. predict
        //
        //for(i=begin; i<end; ++i) {
        //    actualY.push_back(train_set[perm[i]].y);
        //    predictY.push_back(train_set[perm[i]].predict(models));
        //}
        //
        int max_num_xs = *max_element(num_xs.begin(), num_xs.end());
        struct svm_node *x = (struct svm_node*)malloc(sizeof(struct svm_node)*(max_num_xs+1));
        for(i=begin; i<end; ++i) {
            PredictResult val;
            val.y = 0.;
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
                if(!train_set[perm[i]].som.empty())
                    val.som.push_back(train_set[perm[i]].som[j]);
            }
            if(val.y < 0)
                cout << "Warning: sum(10^eachy) out of flow" << endl;
            val.y = log10(val.y);
            actualY.push_back(train_set[perm[i]].y);
            predictY.push_back(val);
        }
        free(x);
        // free
        for(i=0; i<num_types; ++i)
            svm_free_and_destroy_model(&models[i]);
    }
}
*/
/*
float calc_sum_x2(GA1DArrayGenome<float> &g)
{
    float val = 0.;
    for(int i=0; i<g.length()-num_types*3; ++i)
        val += pow(g.gene(i),2);
    return val;
}
float obj_1(vector<double> &actualY, vector<PredictResult> &predictY, GA1DArrayGenome<float> &g)
{
    double mrss = calcRSS(actualY, predictY) / actualY.size();
    return STA_CAST(float, 1./mrss);
}
float obj_2(vector<double> &actualY, vector<PredictResult> &predictY, GA1DArrayGenome<float> &g)
{
    double mrss = calcRSS(actualY, predictY) / actualY.size();
    return STA_CAST(float, 1./mrss+calc_sum_x2(g)/actualY.size());
}
float obj_3(vector<double> &actualY, vector<PredictResult> &predictY, GA1DArrayGenome<float> &g)
{
    double mrss = calcRSS(actualY, predictY) / actualY.size();
    double mauc = 0.;
    int k = 0;
    for(vector<PredictResult>::size_type i=0; i<predictY.size(); ++i) {
        if(!predictY[i].som.empty()) {
            mauc += calcAUC(predictY[i].som, predictY[i].each_y);
            ++k;
        }
    }
    if(mauc == 0.)
        return STA_CAST(float, 1./mrss);
    else {
        mauc /= k;
        return STA_CAST(float, 1./mrss+mauc);
    }
}
float obj_4(vector<double> &actualY, vector<PredictResult> &predictY, GA1DArrayGenome<float> &g)
{
    double mrss = calcRSS(actualY, predictY) / actualY.size();
    double mean_x2 = calc_sum_x2(g) / actualY.size();
    double mauc = 0.;
    int k = 0;
    for(vector<PredictResult>::size_type i=0; i<predictY.size(); ++i) {
        if(!predictY[i].som.empty()) {
            mauc += calcAUC(predictY[i].som, predictY[i].each_y);
            ++k;
        }
    }
    if(mauc == 0.)
        return STA_CAST(float, 1./mrss+mean_x2);
    else {
        mauc /= k;
        return STA_CAST(float, 1./mrss+mean_x2+mauc);
    };
}
//
// IAP = num of ( val+ > val- ) / N+*N-
//
float obj_5(vector<double> &actualY, vector<PredictResult> &predictY, GA1DArrayGenome<float> &g)
{
    double mrss = calcRSS(actualY, predictY) / actualY.size();
    vector<double> iap = calcIAP(actualY, predictY);
    if(iap.empty())
        return STA_CAST(float, 1./mrss);
    else {
        double miap = accumulate(iap.begin(), iap.end(), 0.) / iap.size();
        return STA_CAST(float, 1./mrss+miap);
    }
}
*/
//static int n_calls(0);
static float obj_func(vector<double> &actualY, 
        vector<PredictResult> &predictY, vector<float> &population)
{
    int n = STA_CAST(int, actualY.size());
    double mrss = calcRSS(actualY, predictY) / n;
    double mauc=0., miap=0., mdelta=0., mean_x2=0.;
    if(calc_auc) {
        int k = 0;
        for(vector<PredictResult>::size_type i=0; i<predictY.size(); ++i) {
            if(!predictY[i].som.empty()) {
                mauc += calcAUC(predictY[i].som, predictY[i].each_y);
                ++k;
            }
        }
        mauc /= k;
    }
    if(calc_iap) {
        vector<double> iap = calcIAP(actualY, predictY);
        miap = accumulate(iap.begin(), iap.end(), 0.) / iap.size();
    }
    if(calc_consistency) {
        int k = 0;
        for(vector<PredictResult>::size_type i=0; i<predictY.size(); ++i) {
            for(vector<double>::size_type j=0; j<predictY[i].each_y.size(); ++j) {
                double temp_actual  = actualY[i] + log10(population[k]);
                double temp_predict = predictY[i].each_y[j];
                mdelta += pow(temp_predict-temp_actual, 2);
                ++k;
            }
        }
        mdelta /= k;
    }
    if(calc_x2) {
        int k = 0;
        for(vector<PredictResult>::size_type i=0; i<predictY.size(); ++i) {
            double val=0.;
            for(vector<double>::size_type j=0; j<predictY[i].each_y.size(); ++j) {
                val += pow(population[k],2);
                ++k;
            }
            mean_x2 += val;
        }
        mean_x2 /= predictY.size();
    }
    if(mrss < 1e-3)
        mrss = 1e-3;
    if(mdelta < 1e-3)
        mdelta = 1e-3;
    return STA_CAST(float, 1./mrss+mauc+miap+belta*1./mdelta,10*mean_x2);
}

float myEvaluator(GAGenome &genome)
{
    GA1DArrayGenome<float> &g = DYN_CAST(GA1DArrayGenome<float>&, genome);

    // do cross-validation to estimate the genome
    // 1. randomly split the samples into n folds
    // 2. train `num_types` svm models using {n-1} folds of samples.
    //    for each model, using 5-fold CV to determine the parameter `C,gamma,p`.
    //    during training each model, using log10(y) as y-value
    // 3. applied trained models to the remaining samples
    // 4. calculate 1./RMSE as objective score
    
    vector<double> actualY;
    vector<PredictResult> predictY;
    vector<float> population;
    for(int i=0; i<train_set.count_total_num_atoms()+num_types*3; ++i)
        population.push_back(g.gene(i));
    doCV(nfolds, actualY, predictY, population, true);
    /*
    for(int i=0; i<nfolds; ++i) {
        int begin = i*(train_set.num_samples())/nfolds;
        int end   = (i+1)*(train_set.num_samples())/nfolds;
        do_each(begin, end, actualY, predictY, g);
    }
    */
    return obj_func(actualY, predictY, population);
}


