/*=============================================================================
#     FileName: operators.cpp
#         Desc: 
#       Author: jlpeng
#        Email: jlpeng1201@gmail.com
#     HomePage: 
#      Created: 2014-09-20 10:12:52
#   LastChange: 2015-03-03 15:13:38
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
#include <pthread.h>
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


extern int repeat;
extern int nthread;
extern Sample train_set;
extern int nfolds;
extern vector<vector<svm_problem*> > probs;
extern vector<vector<int> > perm;
extern vector<int> num_xs;
//extern vector<int> num_each_sample;
extern int num_types;
//extern int kernel_type;
//extern vector<vector<vector<double> > > kernel_matrix;
//extern vector<vector<int> > sample_atom_index;
extern vector<svm_parameter*> para;
extern int operator_type;
extern vector<bool> is_som;
//extern float (*obj_func)(vector<double>&, vector<PredictResult>&);
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
float calc_sum_x2(GA1DArrayGenome<float> &g)
{
    float val = 0.;
    for(int i=0; i<g.length()-num_types*3; ++i)
        val += pow(g.gene(i),2);
    return val;
}
float obj_1(vector<double> &actualY, vector<PredictResult> &predictY)
{
    double mrss = calcRSS(actualY, predictY) / actualY.size();
    return STA_CAST(float, 1./mrss);
}
*/
/*
float obj_2(vector<double> &actualY, vector<PredictResult> &predictY, GA1DArrayGenome<float> &g)
{
    double mrss = calcRSS(actualY, predictY) / actualY.size();
    return STA_CAST(float, 1./mrss+calc_sum_x2(g)/actualY.size());
}
*/
/*
float obj_3(vector<double> &actualY, vector<PredictResult> &predictY)
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
*/
/*
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
*/
/*
 * IAP = num of ( val+ > val- ) / N+*N-
 */
/*
float obj_5(vector<double> &actualY, vector<PredictResult> &predictY)
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

class CandidateIndices
{
public:
    CandidateIndices(int n): _n(n), _i(-1) {}
    int next() {
        if(_i < _n-1)
            return ++_i;
        else
            return -1;
    }
private:
    int _n;
    int _i;
};
CandidateIndices *myIndex;
vector<float> obj_values;
vector<float> population;
pthread_mutex_t mut;

// if begin <= end, then use all training samples to train the model, and
// apply the model to training set
static void do_each(int begin, int end, vector<double> &actualY, vector<PredictResult> &predictY, int ii)
{
    int i,j,k,idx_genome;
    for(i=0; i<num_types; ++i)
        probs[ii][i]->l = 0;

    if(para[ii]->kernel_type == PRECOMPUTED) {
        cerr << "Error: PRECOMPUTED kernel not supported!!!" << endl;
        exit(EXIT_FAILURE);
        /*
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
        */
    }
    else {
        // construct train and test set
        idx_genome = 0;
        for(i=0; i<train_set.num_samples(); ++i) {
            // test set
            if(i>=begin && i<end) {
                idx_genome += train_set[perm[ii][i]].num_atoms;
                continue;
            }
            // training set
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
            actualY.push_back(train_set[perm[ii][i]].y);
            predictY.push_back(val);
        }

        // free
        free(x);
        for(i=0; i<num_types; ++i)
            svm_free_and_destroy_model(&models[i]);
    }
    
}

static float obj_func(vector<double> &actualY, vector<PredictResult> &predictY)
{
    int n = STA_CAST(int, actualY.size());
    double mrss = calcRSS(actualY, predictY) / n;
    double mauc=0., miap=0., mdelta=0., mean_x2=0.;
    // `calc_auc` and `calc_iap` are both for estimating if the predicted value
    // is consistent to the observed SOMs
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
    // for each atom site, the `actualY` and `predictY` should be consistency - mrss
    if(calc_consistency) {
        int k = 0;
        for(vector<PredictResult>::size_type i=0; i<predictY.size(); ++i) {
            for(vector<double>::size_type j=0; j<predictY[i].each_y.size(); ++j) {
                double temp_actual  = actualY[i] + log10(population[k]);
                double temp_predict = predictY[i].each_y[j];
                mdelta += pow(temp_predict - temp_actual, 2);
                ++k;
            }
        }
        mdelta /= k;
    }
    if(calc_x2) {
        int k=0;
        for(vector<PredictResult>::size_type i=0; i<predictY.size(); ++i) {
            double val = 0.;
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
    return STA_CAST(float, 1./mrss+mauc+miap+belta*1./mdelta+10.*mean_x2);
}

static void *doCV(void *arg)
{
    int i,j;
    int n = train_set.num_samples();
    while(true) {
        pthread_mutex_lock(&mut);
        i = myIndex->next();
        pthread_mutex_unlock(&mut);
        if(i == -1)
            break;
        vector<double> actualY;
        vector<PredictResult> predictY;
        for(j=0; j<nfolds; ++j) {
            int begin = j*n/nfolds;
            int end   = (j+1)*n/nfolds;
            do_each(begin, end, actualY, predictY, i);
        }
        obj_values[i] = obj_func(actualY, predictY);
    }

    pthread_exit(0);
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
    
    pthread_t *thread = (pthread_t*)malloc(sizeof(pthread_t)*nthread);
    memset(thread, 0, sizeof(pthread_t)*nthread);
    for(int i=0; i<nthread; ++i) {
        int retval = pthread_create(&thread[i], NULL, doCV, NULL);
        if(retval)
            cerr << "Error: failed to create thread " << i+1 << endl;
    }
    for(int i=0; i<nthread; ++i) {
        int retval = pthread_join(thread[i], NULL);
        if(retval)
            cerr << "Error: failed to join thread" << i+1 << endl;
    }
    free(thread);
    delete myIndex;
    
#ifdef TEST_OUTPUT
    cout << "obj_values: ";
    copy(obj_values.begin(), obj_values.end(), ostream_iterator<float>(cout, " "));
    cout << endl;
#endif

    return (accumulate(obj_values.begin(), obj_values.end(), 0.) / repeat);
}
