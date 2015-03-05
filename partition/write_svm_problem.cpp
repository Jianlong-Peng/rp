/*=============================================================================
#     FileName: write_svm_problem.cpp
#         Desc: 
#       Author: jlpeng
#        Email: jlpeng1201@gmail.com
#     HomePage: 
#      Created: 2014-09-23 09:26:55
#   LastChange: 2014-09-24 13:50:45
#      History:
=============================================================================*/
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <string>
#include <utility>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include "tools.h"
#include "../svm/svm.h"
#include <ga/garandom.h>

using namespace std;

void read_train_partition(const char *infile, map<string, map<int, double> > &partition);
void create_write_svm_problems(Sample &train_set, map<string, map<int, double> > &partition, string outname);
void print_null(const char *s) {}



int main(int argc, char *argv[])
{
    if(argc!=3 && argc!=4) {
        cerr << endl << "  Usage: " << argv[0] << "  train_des [train_partition] out_base" << endl
            << endl 
            << "  train_des        : descriptor file" << endl
            << "  [train_partition]: generated by `ga_partition`" << endl
            << "                     if not given, a random y-value will be given" << endl
            << "  out_base         : svm_problem will be writen to {out_base}_x" << endl
            << "                     where, 'x' refers to the atom type, e.g. 0,1,2,3,4,5,6" << endl;
        exit(EXIT_FAILURE);
    }

    svm_set_print_string_function(print_null);

    Sample train_set;
    train_set.read_problem(argv[1]);

    // read train_partition
    map<string, map<int, double> > partition;
    if(argc == 4) {
        read_train_partition(argv[2], partition);
        // create svm_problems
        create_write_svm_problems(train_set, partition, argv[3]);
    }
    else
        create_write_svm_problems(train_set, partition, argv[2]);
    
    return 0;
}

// called by `read_train_partition`
// to extract `atom_id:atom_value`
static void extract_each_value(string &line, string::size_type *i, map<int, double> &each_mol)
{
    string::size_type j = line.find(':', *i);
    int atom_id = atoi(line.substr(*i, j-*i).c_str());
    string::size_type k = line.find(' ', j+1);
    double cl;
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
void read_train_partition(const char *infile, map<string, map<int, double> > &result)
{
    result.clear();
    ifstream inf(infile);
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
        map<int, double> each_mol;
        i++;
        while(i < line.size())
            extract_each_value(line, &i, each_mol);
        result.insert(make_pair(name, each_mol));
    }
    inf.close();

}

void create_write_svm_problems(Sample &train_set, 
        map<string, map<int, double> > &partition, string outname)
{
    for(int i=0; i<train_set.num_samples(); ++i)
        for(int j=0; j<train_set[i].num_atoms; ++j)
            train_set[i].each_y.push_back(partition[train_set[i].name][train_set[i].atom_id[j]]);
    train_set.write_svm_problem(outname);
}
/*
void create_write_svm_problems(Sample &train_set, 
        map<string, map<int, double> > &partition, string outname)
{
    int capacity=50;
    int max_type=-1;
    // 1. count number of Xs for each atom type
    vector<int> num_xs(capacity,0);
    for(int i=0; i<train_set.num_samples(); ++i) {
        for(int j=0; j<train_set[i].num_atoms; ++j) {
            int _type = train_set[i].atom_type[j];
            if(_type >= capacity) {
                capacity *= 2;
                num_xs.resize(capacity,0);
            }
            num_xs[_type] = train_set[i].x[j].size();
            if(_type > max_type)
                max_type = _type;
        }
    }
    // 2. count number of Ys for each atom type
    vector<int> num_each_sample(max_type+1, 0);
    for(int i=0; i<train_set.num_samples(); ++i)
        for(int j=0; j<train_set[i].num_atoms; ++j)
            num_each_sample[train_set[i].atom_type[j]] += 1;
    // 3. create svm_problem for each atom type  -- allocate memory
    vector<svm_problem*> probs(max_type+1, NULL);
    for(int i=0; i<=max_type; ++i) {
        probs[i] = (svm_problem*)malloc(sizeof(svm_problem));
        probs[i]->l = 0;
        probs[i]->y = (double*)malloc(sizeof(double)*(num_each_sample[i]));
        probs[i]->x = (svm_node**)malloc(sizeof(svm_node*)*(num_each_sample[i]));
        for(int j=0; j<num_each_sample[i]; ++j)
            probs[i]->x[j] = (svm_node*)malloc(sizeof(svm_node)*(num_xs[i]+1));
    }
    // -- insert values
    for(int i=0; i<train_set.num_samples(); ++i) {
        for(int j=0; j<train_set[i].num_atoms; ++j) {
            int _type = train_set[i].atom_type[j];
            if(partition.empty())
                probs[_type]->y[probs[_type]->l] = GARandomFloat(0.,1.);
            else
                probs[_type]->y[probs[_type]->l] = partition[train_set[i].name][train_set[i].atom_id[j]];
            for(int k=0; k<num_xs[_type]; ++k) {
                probs[_type]->x[probs[_type]->l][k].index = k+1;
                probs[_type]->x[probs[_type]->l][k].value = train_set[i].x[j][k];
            }
            probs[_type]->x[probs[_type]->l][num_xs[_type]].index = -1;
            probs[_type]->l++;
        }
    }
    // -- check validation
    for(int i=0; i<max_type+1; ++i) {
        if(probs[i]->l != num_each_sample[i]) {
            cerr << "Error: probs[" << i << "]->l =" << probs[i]->l << "  !=  num_each_sample[" << i << "] ="
                << num_each_sample[i] << endl;
            exit(EXIT_FAILURE);
        }
    }

    // 4. output
    for(int i=0; i<=max_type; ++i) {
        ostringstream os;
        os << outname << "_" << i;
        ofstream outf(os.str().c_str());
        for(int j=0; j<probs[i]->l; ++j) {
            outf << probs[i]->y[j];
            for(int k=0; probs[i]->x[j][k].index!=-1; ++k)
                outf << " " << probs[i]->x[j][k].index << ":" << probs[i]->x[j][k].value;
            outf << endl;
        }
        outf.close();
    }

    // free memory
    for(int i=0; i<=max_type; ++i) {
        free(probs[i]->y);
        for(int j=0; j<probs[i]->l; ++j)
            free(probs[i]->x[j]);
        free(probs[i]->x);
        free(probs[i]);
    }

}
*/

