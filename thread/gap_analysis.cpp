/*=============================================================================
#     FileName: main.cpp
#         Desc:
#       Author: jlpeng
#        Email: jlpeng1201@gmail.com
#     HomePage:
#      Created: 2015-03-12 12:23:56
#   LastChange: 2015-03-12 12:47:16
#      History:
=============================================================================*/
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iterator>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include "../utilities/tools.h"
#include "../svm/svm.h"
#include "operators.h"
#include "extern_tools.h"
#include <ga/GA1DArrayGenome.h>
#include <ga/GASStateGA.h>

using namespace std;


string train_des_file("");
string train_som_file("");

Sample test_set;
vector<GA1DArrayGenome<float> > populations;
vector<svm_problem*> myprobs;
svm_parameter* mypara;

// extern variables
Sample train_set;
vector<int> num_xs;
vector<vector<svm_problem*> > probs;
int nfolds(5);
vector<vector<int> > perm;
vector<svm_parameter*> para;
int operator_type(1);
vector<bool> is_som;
int repeat(1);   // do cross-validation {repeat} times
int nthread(1);
bool calc_auc(false);
bool calc_iap(false);
bool calc_consistency(false);
bool calc_x2(false);
double belta(1.);
double wx2(1.);
double wauc(1.);
bool do_log(true);
unsigned actual_seed(0);
int freq_flush(50);

int kernel_type(0);
vector<int> num_each_sample;
int num_types(0);
vector<vector<vector<double> > > kernel_matrix;


void parse_args(const char *infile);
void load_populations(string &infile, int run);
void print_null(const char *s) {}
void init_svm_problems_parameters();
void predict_test_set(Sample &test, int idx_pop);

extern bool cv_detail;

int main(int argc, char *argv[])
{
    if(argc < 7) {
        cerr << endl << "  Usage: " << argv[0] << " [options]" << endl
            << "  [options]" << endl
            << "  --para file: same as input for `gap`" << endl
            << "  --pop  file: log_pop.txt" << endl
            << "  --run   int: specify the populations of which run to be analyzed" << endl
            << "  --test file: test file" << endl
            << "  --seed  int: <default: 0>" << endl
            << endl;
        exit(EXIT_FAILURE);
    }

    string test_file("");
    string pop_file("");
    int run(-1);
    unsigned actual_seed(0);
    int i;
    for(i=1; i<argc; ++i) {
        if(strcmp(argv[i], "--para") == 0)
            parse_args(argv[++i]);
        else if(strcmp(argv[i], "--test") == 0)
            test_file = argv[++i];
        else if(strcmp(argv[i], "--pop") == 0)
            pop_file = argv[++i];
        else if(strcmp(argv[i], "--seed") == 0) {
            istringstream is(argv[++i]);
            is >> actual_seed;
        }
        else if(strcmp(argv[i], "--run") == 0) {
            run = atoi(argv[++i]);
            if(run < 0) {
                cerr << "Error: --run should be >=0, but " << run << " is given" << endl;
                exit(EXIT_FAILURE);
            }
        }
        else {
            cerr << "Error: invalid option " << argv[i];
            exit(EXIT_FAILURE);
        }
    }
    if(i != argc) {
        cerr << "Error: invalid number of arguments" << endl;
        exit(EXIT_FAILURE);
    }
    if(pop_file == "") {
        cerr << "Error: --pop is needed" << endl;
        exit(EXIT_FAILURE);
    }

    svm_set_print_string_function(print_null);

    train_set.read_problem(train_des_file, train_som_file);
    cout << "size of training set: " << train_set.num_samples() << endl
        << "total number of atoms: " << train_set.count_total_num_atoms() << endl;

    test_set.read_problem(test_file);

    // randomize samples for P times !!!!
    GARandomSeed(actual_seed);
    randomize_samples(true);
    construct_svm_problems_parameters();
    init_svm_problems_parameters();

    load_populations(pop_file, run);

    //cv_detail = true;
    for(vector<GA1DArrayGenome<float> >::size_type i=0; i<populations.size(); ++i) {
        cout << endl << "================genome #" << i << "=====================" << endl;
        float val = myEvaluator(populations[i]);
        cout << endl << "  OBJ=" << val << endl;
        cout << endl << "  predicting results on training set:" << endl;
        predict_test_set(train_set, static_cast<int>(i));
        cout << endl << "  predicting results on test set:" << endl;
        predict_test_set(test_set, static_cast<int>(i));
    }

    free_svm_problems_parameters();

    for(int i=0; i<num_types; ++i) {
        free(myprobs[i]->y);
        for(int j=0; j<num_each_sample[i]; ++j)
            free(myprobs[i]->x[j]);
        free(myprobs[i]->x);
        free(myprobs[i]);
    }
    myprobs.clear();
    svm_destroy_param(mypara);
    mypara = NULL;

    return 0;
}

void parse_args(const char *infile)
{
    ifstream inf(infile);
    if(!inf) {
        cerr << "Error: failed to open " << infile << endl;
        exit(EXIT_FAILURE);
    }
    string line;
    string obj_type("000");
    while(getline(inf,line)) {
        if(line.size()==0 || line[0]=='#')
            continue;
        string para;
        string value;
        istringstream is(line);
        is >> para;
        //if(para == "gapara_file")
        //    is >> ga_parameter_file;
        if(para == "train_des")
            is >> train_des_file;
        else if(para == "train_som")
            is >> train_som_file;
        //else if(para == "output")
        //    is >> output_file;
        else if(para == "kernel_type") {
            is >> kernel_type;
            if(kernel_type<0 || kernel_type>2) {
                cerr << "Error: kernel_type should be one of 0,1,2, but " << kernel_type << " is given" << endl;
                exit(EXIT_FAILURE);
            }
            if(kernel_type != 0) {
                cerr << "Error: kernel_type must be 0, but " << kernel_type << " is given" << endl;
                exit(EXIT_FAILURE);
            }
        }
        else if(para == "seed")
            is >> actual_seed;
        else if(para == "freq_flush")
            is >> freq_flush;
        else if(para == "obj_type") {
            is >> obj_type;
            if(obj_type[0] == '1')
                calc_auc = true;
            if(obj_type.size()>=2 && obj_type[1]=='1')
                calc_iap = true;
            if(obj_type.size()>=3 && obj_type[2]=='1')
                calc_consistency = true;
            if(obj_type.size()>=4 && obj_type[3]=='1')
                calc_x2 = true;
       }
       else if(para == "belta")
            is >> belta;
        else if(para == "wx2")
            is >> wx2;
        else if(para == "wauc")
            is >> wauc;
        else if(para == "do_log") {
            int temp;
            is >> temp;
            if(temp == 0)
                do_log = false;
        }
        else if(para == "operator_type")
            is >> operator_type;
        else if(para == "repeat")
            is >> repeat;
        else if(para == "nthread")
            is >> nthread;
        else
            cerr << "Warning: invalid parameter " << para << " being ignored" << endl;
    }

    if(train_des_file == "") {
        cerr << "Error: `train_des` missed" << endl;
        exit(EXIT_FAILURE);
    }
    /*
    if(output_file == "") {
        cerr << "Error: `output` missed" << endl;
        exit(EXIT_FAILURE);
    }
    if(operator_type==2 && train_som_file=="") {
        cerr << "Error: `operator_type=2`, but `train_som` is not given" << endl;
        exit(EXIT_FAILURE);
    }
    if(operator_type!=1 && operator_type!=2) {
        cerr << "Error: `operator_type` can only be 1 or 2, but " << operator_type << " is given" << endl;
        exit(EXIT_FAILURE);
    }
    */
    if(repeat==1 && nthread>1) {
        cerr << "Warning: nthread will be set to be 1 when repeat=1" << endl;
        nthread = 1;
    }
    cout << "train_des: " << train_des_file << endl
        << "train_som: " << train_som_file << endl
        //<< "output file: " << output_file << endl
        << "operator_type: " << operator_type << endl
        << "obj_type: " << obj_type << endl
        << "  calc_auc: " << (calc_auc?"TRUE":"FALSE") << endl
        << "  calc_iap: " << (calc_iap?"TRUE":"FALSE") << endl
        << "  calc_consistency: " << (calc_consistency?"TRUE":"FALSE") << endl
        << "  calc_x2: " << (calc_x2?"TRUE":"FALSE") << endl
        << "belta: " << belta << endl
        << "wx2: " << wx2 << endl
        << "wauc: " << wauc << endl
        << "do_log: " << (do_log?"TRUE":"FALSE") << endl
        << "repeat: " << repeat << endl
        << "nthread: " << nthread << endl;
}    

void load_populations(string &infile, int run)
{
    ifstream inf(infile.c_str());
    if(!inf) {
        cerr << "Error: failed to open " << infile << endl;
        exit(EXIT_FAILURE);
    }
    populations.clear();
    string line;
    int each_run;
    getline(inf, line);
    while(!inf.eof()) {
        istringstream is(line);
        string temp;
        is >> temp >> each_run;
        if(each_run != run) {
            getline(inf,line);
            while(!inf.eof() && line.find("run")!=0)
                getline(inf,line);
        }
        else {
            while(getline(inf,line)) {
                if(line.size()==0 || line.find("run")==0)
                    break;
                istringstream iss(line);
                float val;
                GA1DArrayGenome<float> genome(train_set.count_total_num_atoms()+num_types*3);
                for(int i=0; iss >> val; ++i)
                    genome.gene(i,val);
                populations.push_back(genome);
            }
            break;
        }
    }
    inf.close();
}

void init_svm_problems_parameters()
{
    // num_types & num_xs & number of atoms of type i
    num_types = 0;
    for(int i=0; i<train_set.num_samples(); ++i)
        for(int j=0; j<train_set[i].num_atoms; ++j)
            num_types = (train_set[i].atom_type[j]>num_types)?(train_set[i].atom_type[j]):num_types;
    num_types += 1;
    num_xs.resize(num_types, 0);
    num_each_sample.resize(num_types, 0);
    for(int i=0; i<num_types; ++i) {
         num_xs[i] = 0;
         num_each_sample[i] = 0;
    }
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
    myprobs.resize(num_types);
    for(int i=0; i<num_types; ++i) {
        cout << "for atom type " << i << ", there are " << num_each_sample[i] << " samples and "
            << num_xs[i] << " Xs" << endl;
        myprobs[i] = (svm_problem*)malloc(sizeof(svm_problem));
        myprobs[i]->l = 0;
        myprobs[i]->y = (double*)malloc(sizeof(double)*(num_each_sample[i]));
        myprobs[i]->x = (struct svm_node**)malloc(sizeof(struct svm_node*)*(num_each_sample[i]));
        for(int j=0; j<num_each_sample[i]; ++j) {
            //if(kernel_type != 0)
            //    myprobs[i]->x[j] = (struct svm_node*)malloc(sizeof(struct svm_node)*(num_each_sample[i]+2));
            //else
                myprobs[i]->x[j] = (struct svm_node*)malloc(sizeof(struct svm_node)*(num_xs[i]+1));
        }
    }

    /*
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
    */

    if(mypara != NULL)
        free(mypara);
    mypara = (struct svm_parameter*)malloc(sizeof(struct svm_parameter));
    mypara->svm_type = EPSILON_SVR;
    if(kernel_type != 0)
        mypara->kernel_type = PRECOMPUTED;
    else
        mypara->kernel_type = RBF;
    mypara->degree = 3;
    mypara->gamma = 0.;
    mypara->coef0 = 0;
    mypara->cache_size = 100;
    mypara->eps = 0.001;
    mypara->C = 1.0;
    mypara->nr_weight = 0;
    mypara->weight_label = NULL;
    mypara->weight = NULL;
    mypara->nu = 0.5;
    mypara->p = 0.1;
    mypara->shrinking = 1;
    mypara->probability = 0;

}

void predict_test_set(Sample &test, int idx_pop)
{
    int i,j,k,idx_genome;
    for(i=0; i<num_types; ++i)
        myprobs[i]->l = 0;

    // construct training set
    idx_genome = 0;
    for(i=0; i<train_set.num_samples(); ++i) {
        for(j=0; j<train_set[i].num_atoms; ++j) {
            int _type = train_set[i].atom_type[j];
            for(k=0; k<num_xs[_type]; ++k) {
                myprobs[_type]->x[myprobs[_type]->l][k].index = k+1;
                myprobs[_type]->x[myprobs[_type]->l][k].value = train_set[i].x[j][k];
            }
            myprobs[_type]->x[myprobs[_type]->l][k].index = -1;
            if(do_log)
                myprobs[_type]->y[myprobs[_type]->l] = log10(populations[idx_pop].gene(idx_genome)) + train_set[i].y;
            else
                myprobs[_type]->y[myprobs[_type]->l] = populations[idx_pop].gene(idx_genome)*pow(10,train_set[i].y);
            myprobs[_type]->l++;
            ++idx_genome;
        }
    }
    // construct svm models
    vector<svm_model*> models(num_types, NULL);
    for(i=0; i<num_types; ++i) {
        mypara->C = populations[idx_pop].gene(idx_genome);
        mypara->gamma = populations[idx_pop].gene(idx_genome+1);
        mypara->p = populations[idx_pop].gene(idx_genome+2);
        idx_genome += 3;
        models[i] = svm_train(myprobs[i], mypara);
    }
    // predict
    vector<PredictResult> predictY = test.predict(models, do_log);
    vector<double> actualY;
    for(i=0; i<test.num_samples(); ++i)
        actualY.push_back(test[i].y);
    for(i=0; i<num_types; ++i)
        svm_free_and_destroy_model(&models[i]);

    double rmse = calcRMSE(actualY, predictY);
    double r = calcR(actualY, predictY);
    cout << "  RMSE=" << rmse << endl
        << "     R=" << r << endl
        << endl;
}



