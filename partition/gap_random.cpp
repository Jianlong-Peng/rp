/*=============================================================================
#     FileName: gap_random.cpp
#         Desc: 
#       Author: jlpeng
#        Email: jlpeng1201@gmail.com
#     HomePage: 
#      Created: 2014-10-25 10:19:08
#   LastChange: 2014-10-25 15:15:22
#      History:
=============================================================================*/
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <iterator>
#include <algorithm>
#include <utility>
#include <cstdlib>
#include <cmath>
#include <ga/GA1DArrayGenome.h>
#include <ga/garandom.h>
#include "tools.h"
#include "../svm/svm.h"
#include "extern_tools.h"

using namespace std;

Sample train_set;
Sample test_set;
vector<float> genome;
vector<svm_problem*> probs;
vector<svm_model*> models;
svm_parameter* para;
int num_types;
vector<int> num_xs;
vector<int> num_each_sample;
vector<vector<int> > sample_atom_index;
vector<vector<vector<double> > > kernel_matrix;
int kernel_type(0);
vector<bool> is_som;
vector<int> perm;
int operator_type(2);
int ntimes(1);
bool do_log(true);

void print_null(const char *s){}
void random_init_genome();
void train_models();
void predict(Sample &test, ofstream &outf);

int main(int argc, char *argv[])
{
    if(argc<5 || argc>10) {
        cerr << endl << "  Usage: " << argv[0] << " [options] train_des train_som test_des output" << endl
            << endl << "  [options]" << endl
            << "  --operator int : 1 or 2<default>" << endl
            << "  --times    int : how many times repeat <default: 1>" << endl
            << "  --no-log       : <optional>" << endl
            << "                   if given, models ared trained on y instead of log10(y)" << endl
            << endl;
        exit(EXIT_FAILURE);
    }

    int i;
    for(i=1; i<argc; ++i) {
        if(argv[i][0] != '-')
            break;
        if(strcmp(argv[i], "--operator") == 0) {
            operator_type = atoi(argv[++i]);
            if(operator_type!=1 && operator_type!=2) {
                cerr << "Error: operator_type should be 1 or 2, but " << operator_type << " is given" << endl;
                exit(EXIT_FAILURE);
            }
        }
        else if(strcmp(argv[i], "--times") == 0)
            ntimes = atoi(argv[++i]);
        else if(strcmp(argv[i], "--no-log") == 0)
            do_log = false;
        else {
            cerr << "Error: invalid option " << argv[i] << endl;
            exit(EXIT_FAILURE);
        }
    }
    if(argc-i != 4) {
        cerr << "Error: invalid number of arguments" << endl;
        exit(EXIT_FAILURE);
    }
    string train_des(argv[i]);
    string train_som(argv[i+1]);
    string test_des(argv[i+2]);
    string out_file(argv[i+3]);

    cout << "log10(0)=" << log10(0.) << "; log10(-1.)=" << log10(-1.) << endl;
    cout << "operator_type=" << operator_type << endl;

    svm_set_print_string_function(print_null);

    train_set.read_problem(train_des, train_som);
    test_set.read_problem(test_des);
    int k = 0;
    is_som.resize(train_set.count_total_num_atoms(), false);
    for(i=0; i<train_set.num_samples(); ++i) {
        for(int j=0; j<train_set[i].num_atoms; ++j) {
            if(!train_set[i].som.empty() && train_set[i].som[j]==1)
                is_som[k] = true;
            else
                is_som[k] = false;
            ++k;
        }
    }
    construct_svm_problems_parameters();
    GARandomSeed(0);
    randomize_samples();

    ofstream outf(out_file.c_str());
    if(!outf) {
        cerr << "Error: failed to open " << out_file << endl;
        exit(EXIT_FAILURE);
    }
    outf << "CV\t\ttrain\t\ttest" << endl;
    outf << "rmse\tr\trmse\tr\trmse\tr" << endl;
    vector<double> actualY;
    vector<PredictResult> predictY;
	vector<int> sample_index;
    for(i=0; i<ntimes; ++i) {
        cout << "#iter " << i+1 << endl;

        GARandomSeed(0);
        
        random_init_genome();
    
        doCV(5, actualY, predictY, sample_index, genome, true);
        outf << calcRMSE(actualY, predictY) << "\t" << calcR(actualY, predictY);

        train_models();

        predict(train_set, outf);
        predict(test_set, outf);
        outf << endl;

        for(int i=0; i<num_types; ++i)
            svm_free_and_destroy_model(&models[i]);
    }

    free_svm_problems_parameters();

    return 0;
}

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

void random_init_genome()
{
    genome.resize(train_set.count_total_num_atoms()+3*num_types, 0.);
    int k = 0;
    float temp;
    for(int i=0; i<train_set.num_samples(); ++i) {
        float sum = 0.;
        for(int j=0; j<train_set[i].num_atoms; ++j) {
            if(operator_type == 2) {
                if(is_som[k+j])
                    temp = GARandomFloat(0.5, 1.);
                else
                    temp = GARandomFloat(0., 0.5);
            }
            else
                temp = GARandomFloat(0., 1.);
            sum += temp;
            genome[k+j] = temp;
        }
        for(int j=0; j<train_set[i].num_atoms; ++j)
            genome[k+j] /= sum;
        k += train_set[i].num_atoms;
    }
    for(int i=0; i<num_types; ++i) {
        genome[k] = get_random_c();
        genome[k+1] = get_random_g();
        genome[k+2] = get_random_p();
        k += 3;
    }
    cout << "genome:" << endl;
    copy(genome.begin(), genome.end(), std::ostream_iterator<float>(cout," "));
    cout << endl;
}

void train_models()
{
    int i,j,k,idx_genome;
    for(i=0; i<num_types; ++i)
        probs[i]->l = 0;
    
    idx_genome = 0;
    for(i=0; i<train_set.num_samples(); ++i) {
        for(j=0; j<train_set[i].num_atoms; ++j) {
            int _type = train_set[i].atom_type[j];
            for(k=0; k<num_xs[_type]; ++k) {
                probs[_type]->x[probs[_type]->l][k].index = k+1;
                probs[_type]->x[probs[_type]->l][k].value = train_set[i].x[j][k];
            }
            probs[_type]->x[probs[_type]->l][k].index = -1;
            if(do_log)
                probs[_type]->y[probs[_type]->l] = log10(genome[idx_genome]) + train_set[i].y;
            else
                probs[_type]->y[probs[_type]->l] = genome[idx_genome]*pow(10,train_set[i].y);
            probs[_type]->l++;
            ++idx_genome;
        }
    }

    models.resize(num_types, NULL);
    for(i=0; i<num_types; ++i) {
        para->C = genome[idx_genome];
        para->gamma = genome[idx_genome+1];
        para->p = genome[idx_genome+2];
        idx_genome += 3;
        models[i] = svm_train(probs[i], para);
    }
}

void predict(Sample &test, ofstream &outf)
{
    vector<PredictResult> predictY = test.predict(models,do_log);
    vector<double> actualY;
    for(int i=0; i<test.num_samples(); ++i)
        actualY.push_back(test[i].y);
    /*
    int i,j,k;
    int max_num_xs = *max_element(num_xs.begin(), num_xs.end());
    struct svm_node *x = (struct svm_node*)malloc(sizeof(struct svm_node)*(max_num_xs+1));
    vector<double> actualY;
    vector<double> predictY;
    for(i=0; i<test.num_samples(); ++i) {
        double val = 0.;
        for(j=0; j<test[i].num_atoms; ++j) {
            int _type = test[i].atom_type[j];
            for(k=0; k<num_xs[_type]; ++k) {
                x[k].index = k+1;
                x[k].value = test[i].x[j][k];
            }
            x[k].index = -1;
            double each_value = svm_predict(models[_type], x);
            val += pow(10, each_value);
        }
        actualY.push_back(test[i].y);
        predictY.push_back(log10(val));
    }

    // free
    free(x);
    */
    double rmse = calcRMSE(actualY, predictY);
    double r = calcR(actualY, predictY);
    outf << "\t" << rmse << "\t" << r;
}

