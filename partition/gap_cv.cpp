/*=============================================================================
#     FileName: gap_cv.cpp
#         Desc: 
#       Author: jlpeng
#        Email: jlpeng1201@gmail.com
#     HomePage: 
#      Created: 2014-09-22 14:48:20
#   LastChange: 2014-11-24 16:11:37
#      History:
=============================================================================*/
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <string>
#include <utility>
#include <algorithm>
#include <iterator>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include "../utilities/tools.h"
#include "../svm/svm.h"
#include "extern_tools.h"
#include <ga/garandom.h>

using namespace std;


Sample train_set;
//map<string, map<int, float> > each_atom_y;
vector<float> population;
vector<float> cgp;
vector<svm_problem*> probs;
vector<int> num_xs;
svm_parameter *para(NULL);
vector<int> perm;
int num_types(0);
vector<int> num_each_sample;
int kernel_type(0);
vector<vector<vector<double> > > kernel_matrix;
bool do_log(true);


void read_train_partition(string &infile);
void load_cgp(string &infile);
void print_null(const char *s) {}
//vector<float> construct_population();

int main(int argc, char *argv[])
{
    cout << "CMD:";
    for(int i=0; i<argc; ++i)
        cout << " " << argv[i];
    cout << endl;

    if(argc < 3) {
        cerr << endl << "  Usage: " << argv[0] << " [options] train_des train_partition" << endl
            << endl << "[options]" << endl
            << "  -n int      : <default 5>" << endl
            << "                do {n}-fold CV" << endl
            << "  --repeat int: <default 1>" << endl
            << "                do {n}-fold CV {repeat} times" << endl
            << "                each time, samples will be randomized" << endl
            << "                in this case, `--seed n` will be ignored" << endl
            << "  --cgp file  : where to load `c,g,p` for each atom type instead" << endl
            << "                each line should be `atom_type c g p`" << endl
            << "  --seed n    : specify the random seed" << endl
            << "  --kernel int: <default: 0>" << endl
            << "                0 - RBF kernel" << endl
            << "                1 - tanimoto kernel" << endl
            << "                2 - minMax kernel" << endl
            << "  --verbose int: <default: 0>" << endl
            << "                1 - actual and predicted log(CL) will be shown" << endl
            << "                2 - atom contribution will also be displayed" << endl
            << "  --no-log     : <optional>" << endl
            << "                if given, models will be trained based on y instead of log10(y)" << endl
            << endl
            << "  train_des      : training set" << endl
            << "  train_partition: generated by ga_partition" << endl
            << "                   atom contribution will be read" << endl
            << "                   c,g,p will be used if `--cgp` not given" << endl
            << endl;
        exit(EXIT_FAILURE);
    }

    svm_set_print_string_function(print_null);

    unsigned int seed = 0;
    string cgp_file("");
    int nfolds(5);
    int verbose(0);
    //bool verbose(false);
    int repeat(1);
    int i;
    for(i=1; i<argc; ++i) {
        if(argv[i][0] != '-')
            break;
        if(strcmp(argv[i], "--cgp") == 0)
            cgp_file = argv[++i];
        else if(strcmp(argv[i], "--seed") == 0) {
            istringstream is(argv[++i]);
            is >> seed;
        }
        else if(strcmp(argv[i], "-n") == 0)
            nfolds = atoi(argv[++i]);
        else if(strcmp(argv[i], "--kernel") == 0)
            kernel_type = atoi(argv[++i]);
        else if(strcmp(argv[i], "--verbose") == 0)
            verbose = atoi(argv[++i]);
        else if(strcmp(argv[i], "--repeat") == 0)
            repeat = atoi(argv[++i]);
        else if(strcmp(argv[i], "--no-log") == 0)
            do_log = false;
        else {
            cerr << "Error: invalid option " << argv[i] << endl;
            exit(EXIT_FAILURE);
        }
    }
    if(argc-i != 2) {
        cerr << "Error: invalid number of arguments" << endl;
        exit(EXIT_FAILURE);
    }
    string train_des(argv[i]);
    string train_partition(argv[i+1]);

    if(repeat > 1) {
        if(seed != 0) {
             cout << "Warning: `--repeat " << repeat << "` is given, so `--seed " << seed << "` is ignored" << endl;
             seed = 0;
        }
        if(verbose)
             cout << "Warning: `--repeat " << repeat << "` is given, so `--verbose` is ignored" << endl;
    }

    train_set.read_problem(train_des);

    construct_svm_problems_parameters();

    read_train_partition(train_partition);

    if(cgp_file != "")
        load_cgp(cgp_file);

    int start_cgp = train_set.count_total_num_atoms();
    for(i=0; i<num_types; ++i) {
        cout << "parameter for atom type " << i 
            << ": c=" << population[start_cgp+3*i] << ", g=" << population[start_cgp+3*i+1] 
            << ", p=" << population[start_cgp+3*i+2] << endl;
    }
    
    /*
    vector<float> population = construct_population();
    cout << "run: 0" << endl;
    copy(population.begin(), population.end(), ostream_iterator<float>(cout, " "));
    cout << endl;
    */
    if(repeat > 1) {
        cout << "rmse r" << endl;
        for(i=0; i<repeat; ++i) {
             GARandomSeed();
             randomize_samples(false);
             vector<double> actualY;
             vector<PredictResult> predictY;
             vector<int> sample_index;
             doCV(nfolds, actualY, predictY, sample_index, population, true);
             double rmse = calcRMSE(actualY, predictY);
             double r = calcR(actualY, predictY);
             cout << rmse << " " << r << endl;
        }
    }
    else {
        GARandomSeed(seed);
        cout << "random seed: " << GAGetRandomSeed() << endl;
        randomize_samples();
        
        vector<double> actualY;
        vector<PredictResult> predictY;
		vector<int> sample_index;
        doCV(nfolds, actualY, predictY, sample_index, population, true);
        if(verbose) {
            cout << "name\tactualY\tpredictY";
            if(verbose == 2)
                cout << "\tatom_contribution...";
            cout << endl;
            for(i=0; i<train_set.num_samples(); ++i) {
                cout << train_set[perm[i]].name << "\t" << actualY[i] << "\t" << predictY[i].y;
                if(verbose == 2) {
                    for(vector<double>::iterator iter=predictY[i].each_y.begin(); iter!=predictY[i].each_y.end(); ++iter)
                        cout << "\t" << *iter;
                }
                cout << endl;
            }
        }
        double rmse = calcRMSE(actualY, predictY);
        double r = calcR(actualY, predictY);
        cout << endl << "  result of " << nfolds << "-fold cross-validation" << endl
            << "  RMSE=" << rmse << endl
            << "     R=" << r << endl
            << endl;
    }

    free_svm_problems_parameters();
    
    return 0;
}
/*
// called by `read_train_partition`
// to extract `atom_id:atom_value`
static void extract_each_value(string &line, string::size_type *i, map<int, float> &each_mol)
{
    string::size_type j = line.find(':', *i);
    int atom_id = atoi(line.substr(*i, j-*i).c_str());
    string::size_type k = line.find(' ', j+1);
    float cl;
    if(k == string::npos) {
        cl = atof(line.substr(j+1).c_str());
        *i = line.size();
    }
    else {
        cl = atof(line.substr(j+1, k-j-1).c_str());
        *i = k+1;
    }
    each_mol.insert(make_pair(atom_id, cl));
}
// key=mol_name, value={key=atom_id, value=log10(CL)}
void read_train_partition(string &infile)
{
    each_atom_y.clear();
    cgp.clear();

    ifstream inf(infile.c_str());
    if(!inf) {
        cerr << "Error: failed to read " << infile << endl;
        exit(EXIT_FAILURE);
    }
    string line;
    while(getline(inf, line)) {
        if(line[0] == 'c')
            break;
        string::size_type i = line.find(' ');
        string name = line.substr(0,i);
        map<int, float> each_mol;
        i++;
        while(i < line.size())
            extract_each_value(line, &i, each_mol);
        each_atom_y.insert(make_pair(name, each_mol));
    }
    while(!inf.eof()) {
        string c,g,p;
        istringstream is(line);
        is >> c >> g >> p;
        cgp.push_back(atof(c.substr(2).c_str()));
        cgp.push_back(atof(g.substr(2).c_str()));
        cgp.push_back(atof(p.substr(2).c_str()));
        getline(inf,line);
    }
    inf.close();
}
*/
void read_train_partition(string &infile)
{
    ifstream inf(infile.c_str());
    if(!inf) {
        cerr << "Error: failed to open " << infile << endl;
        exit(EXIT_FAILURE);
    }
    population.clear();
    string line;
    getline(inf,line);
    istringstream is(line);
    float val;
    for(int i=0; i<train_set.count_total_num_atoms(); ++i) {
        is >> val;
        population.push_back(val);
    }
    while(is >> val)
        population.push_back(val);
    inf.close();
}

void load_cgp(string &infile)
{
    ifstream inf(infile.c_str());
    if(!inf) {
        cerr << "Error: failed to open " << infile << endl;
        exit(EXIT_FAILURE);
    }
    string line;
    cgp.clear();
    cgp.resize(num_types*3);
    while(getline(inf,line)) {
        if(line.size()==0 || line[0]=='#')
            continue;
        istringstream is(line);
        int _type;
        float c,g,p;
        is >> _type >> c >> g >> p;
        cgp[3*_type] = c;
        cgp[3*_type+1] = g;
        cgp[3*_type+2] = p;
    }
    inf.close();
    for(vector<float>::size_type i=0; i<cgp.size();++i)
        population.push_back(cgp[i]);
}

/*
vector<float> construct_population()
{
    vector<float> population;
    for(int i=0; i<train_set.num_samples(); ++i)
        for(int j=0; j<train_set[i].num_atoms; ++j)
            population.push_back(each_atom_y[train_set[i].name][train_set[i].atom_id[j]]);
    for(vector<float>::iterator iter=cgp.begin(); iter!=cgp.end(); ++iter)
        population.push_back(*iter);
    return population;
}
*/

