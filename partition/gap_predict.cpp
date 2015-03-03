/*=============================================================================
#     FileName: gap_predict.cpp
#         Desc: 
#       Author: jlpeng
#        Email: jlpeng1201@gmail.com
#     HomePage: 
#      Created: 2014-09-22 09:25:36
#   LastChange: 2015-03-03 16:13:42
#      History:
=============================================================================*/
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cfloat>
#include "svm.h"
#include "tools.h"

using namespace std;

void parse_options(int argc, char *argv[]);
int load_svm_models(vector<svm_model*> &models);
void construct_train_fp(Sample &train_set);
void read_and_predict(vector<svm_model*> &models);
void print_null(const char *s) {}

vector<vector<vector<double> > > train_fp;
int kernel_type(0);
string model_list("");
string test_file("");
string train_file("");
bool detail(false);
string outfile("");
bool do_log(true);

int main(int argc, char *argv[])
{
    if(argc == 1) {
        cerr << endl << "  Usage: " << argv[0] << " [options] output" << endl
            << endl << "[options]" << endl
            << "  --model file: each line should be" << endl
            << "                `type model_file_name`" << endl
            << "                where, `type` referes to atom type, e.g. 0,1,2,3..." << endl
            << "                       `model_file_name` is generated by `gap_train`" << endl
            << "  --test  file: test file of descriptors" << endl
            << "                if actual y-values are given, rmse and r will be calculated" << endl
            << "                (suppose that {y-values} are in `log10`)" << endl
            << "  --kernel int: <optional>" << endl
            << "                only when pre-computed kernel is used" << endl
            << "                1 - tanimoto kernel" << endl
            << "                2 - minMax kernel" << endl
            << "                currently not supported!" << endl
            << "  --train file: <optional>" << endl
            << "                training set. Needed when pre-computed kernels are used" << endl
            << "  --detail    : <optional, default: false>" << endl
            << "                if given, atom contribution will be saved" << endl
            << "  --no-log    : <optional>" << endl
            << "                if given, models were trained based on y instead of log10(y)"<< endl
            << "  output      : where to save predicted results in `log10`" << endl
            << "                each line will be `name [actualY] predictY [atom:cl atom:cl ...]`" << endl
            << endl;
        exit(EXIT_FAILURE);
    }

    //cout << "DBL_MAX=" << DBL_MAX << "; DBL_MIN=" << DBL_MIN << endl;

    svm_set_print_string_function(print_null);

    // parse arguments
    parse_options(argc, argv);
    

    // load svm models
    vector<svm_model*> models;
    int max_type = load_svm_models(models);
    bool has_precompute(false);
    for(int i=0; i<= max_type; ++i) {
        if(models[i]->param.kernel_type == PRECOMPUTED) {
            if(train_file=="") {
                cerr << "Error: pre-computed kernel found but `--train` is not given!" << endl;
                exit(EXIT_FAILURE);
            }
            has_precompute = true;
        }
    }

    if(!has_precompute) {
        if(train_file != "")
            cout << "Warning: no pre-computed kernel found, so `--train` will be ignored!" << endl;
        if(kernel_type != 0)
            cout << "Warning: no pre-computed kernel found, so `--kernel` will be ignored!" << endl;
    }
    else {
        Sample train_set;
        train_set.read_problem(train_file);
        construct_train_fp(train_set);
    }

    // read and predict test samples
    read_and_predict(models);

    for(int i=0; i<=max_type; ++i)
        if(models[i])
            svm_free_and_destroy_model(&models[i]);

    return 0;

}

void construct_train_fp(Sample &train_set)
{
    train_fp.clear();  // vector<vector<vector<double> > >:  [i][][] type i
    int max_type=0;
    for(int i=0; i<train_set.num_samples(); ++i) {
        for(int j=0; j<train_set[i].num_atoms; ++j) {
            if(train_set[i].atom_type[j] > max_type)
                max_type = train_set[i].atom_type[j];
        }
    }
    train_fp.resize(max_type+1);
    for(int i=0; i<train_set.num_samples(); ++i) {
        for(int j=0; j<train_set[i].num_atoms; ++j)
            train_fp[train_set[i].atom_type[j]].push_back(train_set[i].x[j]);
    }
}

void parse_options(int argc, char *argv[])
{
    int i;
    for(i=1; i<argc; ++i) {
        if(argv[i][0] != '-')
            break;
        if(strcmp(argv[i],"--model") == 0)
            model_list = argv[++i];
        else if(strcmp(argv[i],"--test") == 0)
            test_file = argv[++i];
        else if(strcmp(argv[i],"--detail") == 0)
            detail = true;
        else if(strcmp(argv[i], "--train") == 0)
            train_file = argv[++i];
        else if(strcmp(argv[i], "--kernel") == 0) {
            cerr << "Error: currently `--kernel` is not supported" << endl;
            exit(EXIT_FAILURE);
            kernel_type = atoi(argv[++i]);
        }
        else if(strcmp(argv[i], "--no-log") == 0)
            do_log = false;
        else {
            cerr << "Error: invalid option " << argv[i] << endl;
            exit(EXIT_FAILURE);
        }
    }
    if(argc-i != 1) {
        cerr << "Error: invalid number of arguments:";
        for(; i<argc; ++i)
            cerr << " " << argv[i];
        cerr << endl;
        exit(EXIT_FAILURE);
    }
    outfile = argv[i];
    if(model_list.size() == 0) {
        cerr << "Error: '--model' is needed" << endl;
        exit(EXIT_FAILURE);
    }
    if(test_file.size() == 0) {
        cerr << "Error: '--test' is needed" << endl;
        exit(EXIT_FAILURE);
    }
    if(kernel_type<0 || kernel_type>2) {
        cerr << "Error: kernel_type should be either 1 or 2, or not specified" << endl;
        exit(EXIT_FAILURE);
    }
}

int load_svm_models(vector<svm_model*> &models)
{
    ifstream inf(model_list.c_str());
    if(!inf) {
        cerr << "Error: failed to open " << model_list << endl;
        exit(EXIT_FAILURE);
    }
    int capacity(10);
    int max_type(-1);
    int _type;
    string model_name;
    models.clear();
    models.resize(capacity, NULL);
    string line;
    while(getline(inf, line)) {
        if(line.size()==0 || line[0]=='#')
            continue;
        istringstream is(line);
        is >> _type >> model_name;
        if(_type >= capacity) {
            capacity *= 2;
            models.resize(capacity, NULL);
        }
        if(_type > max_type)
            max_type = _type;
        models[_type] = svm_load_model(model_name.c_str());
    }
    inf.close();

    return max_type;
}

double calcKernel(vector<double> &x1, vector<double> &x2)
{
    if(kernel_type == 1)
        return tanimotoKernel(x1, x2);
    else if(kernel_type == 2)
        return minMaxKernel(x1,x2);
    else {
        cerr << "Error: in calcKernel, kernel_type should be either 1 or 2, but " << kernel_type << " is given" << endl;
        exit(EXIT_FAILURE);
    }
}

void read_and_predict(vector<svm_model*> &models)
{
    ofstream outf(outfile.c_str());
    if(!outf) {
        cerr << "Error: failed to opne " << outfile << endl;
        exit(EXIT_FAILURE);
    }
    
    Sample test_set;
    test_set.read_problem(test_file);
    vector<PredictResult> predictY = test_set.predict(models,do_log);
    vector<double> actualY;
    bool hasy(true);
    for(int i=0; i<test_set.num_samples(); ++i) {
        outf << test_set[i].name;
        if(test_set[i].hasy) {
            actualY.push_back(test_set[i].y);
            outf << " " << test_set[i].y;
        }
        else
            hasy = false;
        outf << " " << predictY[i].y;
        if(detail) {
            for(int j=0; j<test_set[i].num_atoms; ++j)
                outf << " " << test_set[i].atom_id[j] << ":" << predictY[i].each_y[j];
        }
        outf << endl;
    }
    outf.close();
    
    if(hasy) {
        double rmse = calcRMSE(actualY, predictY);
        double r = calcR(actualY, predictY);
        cout << endl 
            << "  totally read and predict " << test_set.num_samples() << " test samples" << endl
            << "  and predicted results are saved in " << outfile << endl << endl
            << "  RMSE=" << rmse << endl
            << "     R=" << r << endl
            << endl;
    }
}
/*
void read_and_predict(vector<svm_model*> &models)
{
    ifstream inf(test_file.c_str());
    if(!inf) {
        cerr << "Error: failed to open " << test_file << endl;
        exit(EXIT_FAILURE);
    }
    ofstream outf(outfile.c_str());
    if(!outf) {
        cerr << "Error: failed to opne " << outfile << endl;
        exit(EXIT_FAILURE);
    }

    int line_no(0);
    int num_samples(0);
    string line;
    string name;
    bool hasy(true);
    vector<double> actualY;
    vector<double> predictY;
    getline(inf,line);
    while(!inf.eof()) {
        ++line_no;
        ++num_samples;
        string::size_type j = line.find("\t");
        if(j == string::npos) {
            name = line;
            hasy = false;
        }
        else {
            name = line.substr(0, j);
            actualY.push_back(atof(line.substr(j+1).c_str()));
        }
        vector<int> atom_ids;
        vector<double> each_ys;
        bool invalid(false);
        // read and predict each atom
        while(getline(inf, line) && line[0]=='\t') {
            ++line_no;
            string::size_type k = line.find(':');
            atom_ids.push_back(atoi(line.substr(1,k).c_str()));
            string::size_type kk = line.find(':', k+1);
            int atom_type = atoi(line.substr(k+1, kk-k-1).c_str());
            if(models[atom_type] == NULL) {
                cerr << "Error: invalid atom type " << atom_type << " found in line " << line_no << endl
                    << "       there is no corresponding model loaded from `model_list` file" << endl;
                invalid = true;
                break;
            }
            string::size_type kkk = kk + 1;
            vector<double> xs;
            while(kkk < line.size()) {
                while(kkk<line.size() && line[kkk]!=',')
                    ++kkk;
                xs.push_back(atof(line.substr(kk+1, kkk-kk-1).c_str()));
                kk = kkk;
                ++kkk;
            }
            svm_node *svm_xs;
            if(models[atom_type]->param.kernel_type == PRECOMPUTED) {
                svm_xs = (svm_node*)malloc(sizeof(svm_node)*(train_fp[atom_type].size()+2));
                svm_xs[0].index = 0;
                for(vector<vector<double> >::size_type i=0; i<train_fp[atom_type].size(); ++i) {
                    svm_xs[i+1].index = i+1;
                    svm_xs[i+1].value = calcKernel(xs, train_fp[atom_type][i]);
                }
                svm_xs[train_fp[atom_type].size()+1].index = -1;
            }
            else {
                svm_xs = (svm_node*)malloc(sizeof(svm_node)*(xs.size()+1));
                for(vector<double>::size_type i=0; i<xs.size(); ++i) {
                    svm_xs[i].index = i+1;
                    svm_xs[i].value = xs[i];
                }
                svm_xs[xs.size()].index = -1;
            }
            each_ys.push_back(svm_predict(models[atom_type], svm_xs));
            free(svm_xs);
        }
        // 
        if(invalid) {
            while(getline(inf,line) && line[0]=='\t')
                ++line_no;
            outf << name;
            if(hasy)
                outf << " " << actualY.back();
            outf << " ERROR" << endl;
            actualY.pop_back();
            --num_samples;
        }
        else {
            outf << name;
            if(hasy)
                outf << " " << actualY.back();
            double pred = 0.;
            for(vector<double>::iterator iter=each_ys.begin(); iter!=each_ys.end(); ++iter)
                pred += pow(10, *iter);
            pred = log10(pred);
            predictY.push_back(pred);
            outf << " " << pred;
            if(detail) {
                for(vector<int>::size_type i=0; i<atom_ids.size(); ++i)
                    outf << " " << atom_ids[i] << ":" << each_ys[i];
            }
            outf << endl;
        }
    }

    inf.close();
    outf.close();

    if(hasy) {
        double rmse = calcRMSE(actualY, predictY);
        double r = calcR(actualY, predictY);
        cout << endl 
            << "  totally read and predict " << num_samples << " test samples" << endl
            << "  and predicted results are saved in " << outfile << endl << endl
            << "  RMSE=" << rmse << endl
            << "     R=" << r << endl
            << endl;
    }
}
*/
