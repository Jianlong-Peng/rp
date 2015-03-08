/*=============================================================================
#     FileName: gap_analysis.cpp
#         Desc: 
#       Author: jlpeng
#        Email: jlpeng1201@gmail.com
#     HomePage: 
#      Created: 2014-10-22 14:54:03
#   LastChange: 2014-11-24 07:11:47
#      History:
=============================================================================*/
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <iterator>
#include <algorithm>
#include <numeric>
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
int nfolds(5);
vector<svm_problem*> probs;
vector<int> perm;
vector<int> num_xs;
vector<int> num_each_sample;
int num_types;
int operator_type(2);
int kernel_type(0);
vector<vector<vector<double> > > kernel_matrix;
vector<vector<int> > sample_atom_index;
svm_parameter *para;


bool calc_auc(false);
bool calc_iap(false);
bool calc_consistency(false);
bool calc_x2(false);
int run(-1);
vector<vector<float> > population;
bool do_log(true);

void print_null(const char *s) {}
void read_population(string &pop_file);
void predict_test_set(Sample &test, int idx_pop);
void write_train_partition();
float obj_func(vector<double> &actualY,
    vector<PredictResult> &predictY, vector<float> &population);

int main(int argc, char *argv[])
{
    cout << "CMD:";
    for(int i=0; i<argc; ++i)
        cout << " " << argv[i];
    cout << endl;

    if(argc < 7) {
        cerr << endl << "  Usage: " << argv[0] << " [options]" << endl
            << endl << "  [options]" << endl
            << "  --train file : training set" << endl
            << "  --som   file : SOM for training set <optional>" << endl
            << "  --test  file : test set <optional>" << endl
            << "  --pop   file : population file log_pop.txt" << endl
            << "  --run   int  : specify which run of population should be used" << endl
            << "  --seed  int  : random seed <default: 0>" << endl
            << "  --nfold int  : do n-fold cross validation <default: 5>" << endl
            << "  --kernel type: <default 0>" << endl
            << "    0 - RBF" << endl
            << "    1 - tanimoto kernel" << endl
            << "    2 - minMax kernel" << endl
            << "  --obj   str  : <default: '0000'>" << endl
            << "                 same as specified in `para.txt`" << endl
            << "  --no-log     : <optional>" << endl
            << "    if given, models will be trained on y instead of log10(y)" << endl
            << endl;
        exit(EXIT_FAILURE);
    }

    string train_file("");
    string som_file("");
    string test_file("");
    string pop_file("");
    string obj_type("000");
    unsigned seed(0);
    int i;
    for(i=1; i<argc; ++i) {
        if(strcmp(argv[i], "--train") == 0)
            train_file = argv[++i];
        else if(strcmp(argv[i], "--som") == 0)
            som_file = argv[++i];
        else if(strcmp(argv[i], "--test") == 0)
            test_file = argv[++i];
        else if(strcmp(argv[i], "--pop") == 0)
            pop_file = argv[++i];
        else if(strcmp(argv[i], "--seed") == 0) {
            istringstream is(argv[++i]);
            is >> seed;
        }
        else if(strcmp(argv[i], "--kernel") == 0) {
            kernel_type = atoi(argv[++i]);
            if(kernel_type<0 || kernel_type>2) {
                cerr << "Error: kernel_type should be one of `0,1,2`, but " << kernel_type << " is given" << endl;
                exit(EXIT_FAILURE);
            }
        }
        else if(strcmp(argv[i], "--run") == 0) {
            run = atoi(argv[++i]);
            if(run < 0) {
                cerr << "Error: --run should be >=0, but " << run << " is given" << endl;
                exit(EXIT_FAILURE);
            }
        }
        else if(strcmp(argv[i], "--nfold") == 0)
            nfolds = atoi(argv[++i]);
        else if(strcmp(argv[i], "--obj") == 0)
            obj_type = argv[++i];
        else if(strcmp(argv[i], "--no-log") == 0)
            do_log = false;
        else {
            cerr << "Error: invalid option " << argv[i];
            exit(EXIT_FAILURE);
        }
    }
    if(train_file == "") {
        cerr << "Error: --train is needed" << endl;
        exit(EXIT_FAILURE);
    }
    if(pop_file == "") {
        cerr << "Error: --pop is needed" << endl;
        exit(EXIT_FAILURE);
    }
    if(obj_type[0] == '1')
        calc_auc = true;
    if(obj_type.size()>=2 && obj_type[1]=='1')
        calc_iap = true;
    if(obj_type.size()>=3 && obj_type[2]=='1')
        calc_consistency = true;
    if(obj_type.size()>=4 && obj_type[3]=='1')
        calc_x2 = true;
    
    cout << "CMD:";
    for(int i=0; i<argc; ++i)
        cout << " " << argv[i];
    cout << endl;
    cout << "obj: " << obj_type << endl
        << "  calc_auc: " << ((calc_auc)?"TRUE":"FALSE") << endl
        << "  calc_iap: " << ((calc_iap)?"TRUE":"FALSE") << endl
        << "  calc_consistency: " << ((calc_consistency)?"TRUE":"FALSE") << endl
        << "do_log: " << (do_log?"TRUE":"FALSE") << endl;

    svm_set_print_string_function(print_null);

    train_set.read_problem(train_file, som_file);
    
    read_population(pop_file);
    if(population.empty()) {
        cerr << "Error: failed to load population from " << pop_file << endl;
        exit(EXIT_FAILURE);
    }

    GARandomSeed(seed);
    randomize_samples();
    construct_svm_problems_parameters();

    write_train_partition();

    if(test_file != "")
        test_set.read_problem(test_file);

    cout << "OBJ = 1/mrss + mauc + miap + 1/mdelta + 10*mean_x2" << endl;
    for(vector<vector<float> >::size_type i=0; i<population.size(); ++i) {
        cout << endl << "================genome #" << i << "=====================" << endl;
        vector<double> actualY;
        vector<PredictResult> predictY;
        doCV(nfolds, actualY, predictY, population[i], true);
        double rmse = calcRMSE(actualY, predictY);
        double r = calcR(actualY, predictY);
        cout << endl << "  result of " << nfolds << "-fold cross-validation" << endl
            << "  RMSE=" << rmse << endl
            << "     R=" << r << endl
            << "   OBJ=" << obj_func(actualY,predictY, population[i]) << endl
            << endl;
        cout << endl << "  predicting results on training set:" << endl;
        predict_test_set(train_set, static_cast<int>(i));
        if(test_file != "") {
            cout << endl << "  predicting results on test set:" << endl;
            predict_test_set(test_set, static_cast<int>(i));
        }
    }

    free_svm_problems_parameters();

    return 0;
}

static bool startswith(string &line, const char *word)
{
    size_t len = strlen(word);
    if(line.size() < len)
        return false;
    if(line.substr(0,len) == word)
        return true;
    else
        return false;
}
void read_population(string &pop_file)
{
    ifstream inf(pop_file.c_str());
    if(!inf) {
        cerr << "Error: failed to open " << pop_file << endl;
        exit(EXIT_FAILURE);
    }
    string line;
    int each_run;
    getline(inf, line);
    while(!inf.eof()) {
        istringstream is(line);
        string temp;
        is >> temp >> each_run;
        if(each_run != run) {
            getline(inf,line);
            while(!inf.eof() && !startswith(line, "run"))
                getline(inf,line);
        }
        else {
            while(getline(inf,line)) {
                if(line.size()==0 || startswith(line, "run"))
                    break;
                istringstream iss(line);
                float val;
                vector<float> genome;
                for(int i=0; iss >> val; ++i)
                    genome.push_back(val);
                population.push_back(genome);
            }
            break;
        }
    }
    inf.close();
}

void write_train_partition()
{
    for(vector<vector<float> >::size_type i=0; i<population.size(); ++i) {
        int idx = 0;
        cout << endl << "genome #" << i << endl;
        for(int j=0; j<train_set.num_samples(); ++j) {
            cout << train_set[j].name;
            for(int k=0; k<train_set[j].num_atoms; ++k) {
                cout << " " << train_set[j].atom_id[k] << ":" << log10(population[i][idx]) + train_set[j].y;
                ++idx;
            }
            cout << endl;
        }
        cout << "num_types=" << num_types << endl;
        for(int k=0; k<num_types; ++k) {
            cout << "c=" << population[i][idx] << " g=" << population[i][idx+1] << " p=" << population[i][idx+2] << endl;
            idx += 3;
        }
    }
}

void predict_test_set(Sample &test, int idx_pop)
{
    int i,j,k,idx_genome;
    for(i=0; i<num_types; ++i)
        probs[i]->l = 0;

    // construct training set
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
                probs[_type]->y[probs[_type]->l] = log10(population[idx_pop][idx_genome]) + train_set[i].y;
            else
                probs[_type]->y[probs[_type]->l] = population[idx_pop][idx_genome]*pow(10,train_set[i].y);
            probs[_type]->l++;
            ++idx_genome;
        }
    }
    // construct svm models
    vector<svm_model*> models(num_types, NULL);
    for(i=0; i<num_types; ++i) {
        para->C = population[idx_pop][idx_genome];
        para->gamma = population[idx_pop][idx_genome+1];
        para->p = population[idx_pop][idx_genome+2];
        idx_genome += 3;
        models[i] = svm_train(probs[i], para);
    }
    // predict
    vector<PredictResult> predictY = test.predict(models, do_log);
    vector<double> actualY;
    for(i=0; i<test.num_samples(); ++i)
        actualY.push_back(test[i].y);
    /*
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
    for(i=0; i<num_types; ++i)
        svm_free_and_destroy_model(&models[i]);
    
    double rmse = calcRMSE(actualY, predictY);
    double r = calcR(actualY, predictY);
    cout << "  RMSE=" << rmse << endl
        << "     R=" << r << endl
        << endl;
}

float obj_func(vector<double> &actualY,
    vector<PredictResult> &predictY, vector<float> &population)
{
    int n = static_cast<int>(actualY.size());
    double mrss = calcRSS(actualY, predictY) / n;
    double mauc=0., miap=0., mdelta=0., mean_x2=0.;
    if(calc_auc) {
        int k=0;
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
            double val = 0.;
            for(vector<double>::size_type j=0; j<predictY[i].each_y.size(); ++j) {
                val += pow(population[k],2);
                ++k;
            }
            mean_x2 += val;
        }
        mean_x2 /= predictY.size();
    }
    cout << "mrss=" << mrss << " mauc=" << mauc << " miap=" << miap << " mdelta=" << mdelta << " mean_x2=" << mean_x2 << endl;
    if(mrss < 1e-3)
        mrss = 1e-3;
    if(mdelta < 1e-3)
        mdelta = 1e-3;
    return STA_CAST(float, 1./mrss+mauc+miap+1./mdelta+10*mean_x2);
}

