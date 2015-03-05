/*=============================================================================
#     FileName: tools.cpp
#         Desc: 
#       Author: jlpeng
#        Email: jlpeng1201@gmail.com
#     HomePage: 
#      Created: 2014-09-15 19:57:13
#   LastChange: 2015-03-03 16:00:44
#      History:
=============================================================================*/
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <set>
#include <cstdlib>
#include <cmath>
#include "tools.h"
#include "../svm/svm.h"

using std::cout;
using std::cerr;
using std::endl;
using std::vector;
using std::string;
using std::ifstream;
using std::ofstream;
using std::ostringstream;
using std::set;


PredictResult Molecule::predict(vector<svm_model*> &models, bool do_log)
{
    PredictResult val;
    double each_value;
    val.y = 0.;
    for(int j=0; j<num_atoms; ++j) {
        int _type = atom_type[j];
        if(models[_type]->param.kernel_type == PRECOMPUTED) {
            cerr << "Error(Molecule::predict): it's not ready for PRECOMPUTED kernel!!" << endl;
            exit(EXIT_FAILURE);
        }
        else {
            struct svm_node *nodex = (struct svm_node*)malloc(sizeof(struct svm_node)*(x[j].size()+1));
            vector<double>::size_type k;
            for(k=0; k<x[j].size(); ++k) {
                nodex[k].index = k+1;
                nodex[k].value = x[j][k];
            }
            nodex[k].index = -1;
            each_value = svm_predict(models[_type], nodex);
#ifdef DEBUG
            cout << "  each_value=" << each_value << "  (atom id=" << atom_id[j] << ")" << endl;
#endif
            free(nodex);
        }
        if(do_log) {
            this->each_y.push_back(each_value);
            val.each_y.push_back(each_value);
            val.y += pow(10, each_value);
        }
        else {
            this->each_y.push_back(log10(each_value));
            val.each_y.push_back(log10(each_value));
            val.y += each_value;
        }
        if(!som.empty())
            val.som.push_back(som[j]);
    }
    if(val.y < 0.)
        cout << "Warning(Molecule::predict): sum(10^eachy) overflow!!!" << endl;
    val.y = log10(val.y);
#ifdef DEBUG
    cout << "predicted log10(CLint)=" << val.y << endl;
#endif
        
    return val;
}

PredictResult Molecule::predict(vector<svm_model*> &models, Sample &train, 
        double (*calcKernel)(vector<double> &x, vector<double> &y), bool do_log)
{
    PredictResult val;
    
    for(int i=0; i<this->num_atoms; ++i) {
        int _type = this->atom_type[i];
        if(models[_type]->param.kernel_type != PRECOMPUTED) {
            cerr << "Error: PRECOMPUTED kernel is needed!!!!!" << endl;
            exit(EXIT_FAILURE);
        }
        int n = train.number_atoms_of_type(this->atom_type[i]);
        struct svm_node *nodex = (struct svm_node*)malloc(sizeof(struct svm_node)*(n+2));
        nodex[0].index = 0;
        nodex[0].value = i+1;
        int z=1;
        for(int j=0; j<train.num_samples(); ++j) {
            for(int k=0; k<train[j].num_atoms; ++k) {
                if(train[j].atom_type[k] != this->atom_type[i])
                    continue;
                nodex[z].index = z;
                nodex[z].value = calcKernel(this->x[i], train[j].x[k]);
                ++z;
            }
        }
        double each_value = svm_predict(models[_type], nodex);
        if(do_log) {
            this->each_y.push_back(each_value);
            val.each_y.push_back(each_value);
            val.y += pow(10, each_value);
        }
        else {
            this->each_y.push_back(log10(each_value));
            val.each_y.push_back(log10(each_value));
            val.y += each_value;
        }
        if(!((this->som).empty()))
            val.som.push_back(this->som[i]);
        free(nodex);
    }
    if(val.y < 0.) {
        cout << "Warning(Molecule::predict): sum(10^eachy) overflow!!!" << endl;
    }
    val.y = log10(val.y);
    
    return val;
}

void Sample::read_problem(string train_des_file, string train_som_file)
{
    ifstream inf1(train_des_file.c_str());
    if(!inf1) {
        cerr << "Error: failed to open file " << train_des_file << endl;
        exit(EXIT_FAILURE);
    }
    this->som = false;
    // 1. count number of samples
    int num_samples = 0;
    string line;
    while(getline(inf1,line)) {
        if(line.size()!=0 && line[0]!='\t')
            ++num_samples;
    }
    inf1.close();
    data.resize(num_samples);
    // 2. go back to the begining of file and read data
    ifstream inf2(train_des_file.c_str());
    int i=0;
    getline(inf2,line);
    while(!inf2.eof()) {
        string::size_type j = line.find("\t");
        if(j == string::npos) {
            data[i].name = line;
            data[i].hasy = false;
        }
        else {
            data[i].name = line.substr(0,j);
            data[i].y = atof(line.substr(j+1).c_str());
            data[i].hasy = true;
        }
        getline(inf2,line);
        // parse atoms
        while(!inf2.eof() && line[0]=='\t') {
            data[i].num_atoms += 1;
            // get atom id
            string::size_type j = line.find(":");
            data[i].atom_id.push_back(atoi(line.substr(1,j).c_str()));
            // get atom type
            string::size_type jj = line.rfind(":");
            data[i].atom_type.push_back(atoi(line.substr(j+1,jj-j-1).c_str()));
            // extract x-values
            string::size_type jjj = jj + 1;
            data[i].x.push_back(vector<double>());
            while(jjj < line.size()) {
                while(jjj<line.size() && line[jjj]!=',')
                    ++jjj;
                data[i].x.back().push_back(atof(line.substr(jj+1,jjj-jj-1).c_str()));
                jj = jjj;
                ++jjj;
            }
            // next atom
            getline(inf2,line);
        }
        // next molecule
        ++i;
    }
    inf2.close();
    //~~~~~~~~~~~~~~~read SOMs~~~~~~~~~~~~~~~~~~~~~~~~~
    if(train_som_file.size() == 0)
        return ;
    ifstream inf3(train_som_file.c_str());
    if(!inf3) {
        cerr << "Error: failed to open file " << train_som_file << endl;
        exit(EXIT_FAILURE);
    }
    getline(inf3,line);
    i = 0;
    while(getline(inf3,line)) {
        int count = -2;
        string::size_type tab_start = 0;
        for(string::size_type j=0; j<line.size(); ++j) {
            if(line[j] == '\t') {
                ++count;
                if(count == 1)
                    tab_start = j;
            }
        }
        if(count == 0) {
            ++i;
            continue;
        }
        data[i].som.resize(data[i].num_atoms, false);
        bool bond(false);
        for(string::size_type j=tab_start+1; j<line.size(); ++j) {
            if(line[j] == '-')
                bond = true;
            if(line[j] == '\t') {
                if(!bond) {
                    int atom_id = atoi(line.substr(tab_start+1, j-tab_start-1).c_str());
                    int index;
                    for(index=0; index<data[i].num_atoms; ++index)
                        if(data[i].atom_id[index] == atom_id)
                            break;
                    if(index != data[i].num_atoms)
                        data[i].som[index] = true;
                }
                tab_start = j;
                bond = false;
            }
        }
        if(!bond) {
            int atom_id = atoi(line.substr(tab_start+1, line.size()-tab_start-1).c_str());
            int index;
            for(index=0; index<data[i].num_atoms; ++index)
                if(data[i].atom_id[index] == atom_id)
                    break;
            if(index != data[i].num_atoms)
                data[i].som[index] = false;
        }
        ++i;
    }
    inf3.close();
    this->som = true;
    
}

void Sample::write_problem(string outfile)
{
    ofstream outf(outfile.c_str());
    if(!outf) {
        cerr << "Error: failed to open " << outfile << endl;
        exit(EXIT_FAILURE);
    }
    
    for(vector<Molecule>::size_type i=0; i<data.size(); ++i) {
        outf << data[i].name;
        if(data[i].hasy)
            outf << "\t" << data[i].y;
        outf << endl;
        for(int j=0; j<data[i].num_atoms; ++j) {
            outf << "\t" << data[i].atom_id[j] << ":" << data[i].atom_type[j] << ":" << data[i].x[j][0];
            for(vector<double>::size_type k=1; k<data[i].x[j].size(); ++k)
                outf << "," << data[i].x[j][k];
            outf << endl;
        }
    }
    outf.close();
}

void Sample::write_svm_problem(string outfile, int type)
{
    ofstream outf(outfile.c_str());
    if(!outf) {
        cerr << "Error(Sample::write_svm_problem): failed to open " << outfile << endl;
        exit(EXIT_FAILURE);
    }
    for(vector<Molecule>::size_type i=0; i<data.size(); ++i) {
        for(int j=0; j<data[i].num_atoms; ++j) {
            if(data[i].atom_type[j] != type)
                continue;
            if(data[i].each_y.empty())
                outf << "0";
            else
                outf << data[i].each_y[j];
            for(vector<double>::size_type k=0; k<data[i].x[j].size(); ++k)
                outf << " " << k+1 << ":" << data[i].x[j][k];
            outf << endl;
        }
    }
    outf.close();
}

void Sample::write_svm_problem(string outfile)
{
    set<int> all_types;
    for(vector<Molecule>::size_type i=0; i<data.size(); ++i)
        for(int j=0; j<data[i].num_atoms; ++j)
            all_types.insert(data[i].atom_type[j]);
    for(set<int>::iterator iter=all_types.begin(); iter!=all_types.end(); ++iter) {
        ostringstream os;
        os << outfile << "_" << *iter;
        write_svm_problem(os.str(), *iter);
    }
}

vector<PredictResult> Sample::predict(vector<svm_model*> &models, bool do_log)
{
    vector<PredictResult> predictY;
    for(vector<Molecule>::size_type i=0; i<data.size(); ++i) {
#ifdef DEBUG
        cout << ">>> " << data[i].name << "\t" << data[i].y << endl;
#endif
        predictY.push_back(data[i].predict(models,do_log));
    }
    
    return predictY;    
}

vector<PredictResult> Sample::predict(vector<svm_model*> &models, Sample &train, 
        double (*calcKernel)(vector<double> &x, vector<double> &y), bool do_log)
{
    vector<PredictResult> predictY;
    for(vector<Molecule>::size_type i=0; i<data.size(); ++i)
        predictY.push_back(data[i].predict(models, train, calcKernel, do_log));
    
    return predictY;   
}

int Sample::number_atoms_of_type(int type)
{
    int num = 0;
    for(vector<Molecule>::size_type i=0; i<data.size(); ++i) {
        for(int j=0; j<data[i].num_atoms; ++j) {
            if(data[i].atom_type[j] == type)
                ++num;
        }
    }
    return num;
}

class Comp
{
private:
    const vector<double> &values;
public:
    Comp(const vector<double> &ptr): values(ptr) {}
    bool operator()(int i, int j) const {
        return values[i] > values[j];
    }
};

double calcAUC(const vector<bool> &labels, const vector<double> &ys)
{
    double roc = 0;
    vector<int> indices(labels.size());
    for(int i=0; i<static_cast<int>(labels.size()); ++i)
        indices[i] = i;

    std::sort(indices.begin(), indices.end(), Comp(ys));
    
    int tp=0, fp=0;
    for(vector<int>::size_type i=0; i<labels.size(); ++i) {
        if(labels[indices[i]])
            ++tp;
        else {
            roc += tp;
            ++fp;
        }
    }

    if(tp==0 || fp==0) {
        //cerr << "warning: Too few postive true labels or negative true labels" << endl;
        roc = 0.;
    }
    else
        roc = roc / tp / fp;

    return roc;
}

double calcRSS(const vector<double> &actualY, const vector<double> &predictY)
{
    double val = 0.;
    for(vector<double>::size_type i=0; i<actualY.size(); ++i)
        val += pow(actualY[i]-predictY[i],2);
    return val;
}
double calcRSS(const vector<double> &actualY, const vector<PredictResult> &predictY)
{
    double val = 0.;
    for(vector<double>::size_type i=0; i<actualY.size(); ++i)
        val += pow(actualY[i]-predictY[i].y,2);
    return val;
}
double calcRMSE(const vector<double> &actualY, const vector<double> &predictY)
{
    return sqrt(calcRSS(actualY, predictY) / actualY.size());
}
double calcRMSE(const vector<double> &actualY, const vector<PredictResult> &predictY)
{
    return sqrt(calcRSS(actualY, predictY) / actualY.size());
}
double calcR(const vector<double> &actualY, const vector<double> &predictY)
{
    double sumx=0., sumy=0., sumxy=0., sumxx=0., sumyy=0.;
    for(vector<double>::size_type i=0; i<actualY.size(); ++i) {
        sumx  += actualY[i];
        sumy  += predictY[i];
        sumxy += actualY[i]*predictY[i];
        sumxx += actualY[i]*actualY[i];
        sumyy += predictY[i]*predictY[i];
    }
    int n = static_cast<int>(actualY.size());
    double meanx = sumx/n;
    double meany = sumy/n;
    double r = (sumxy-n*meanx*meany) / sqrt((sumxx-n*meanx*meanx)*(sumyy-n*meany*meany));
    return r;
}
double calcR(const vector<double> &actualY, const vector<PredictResult> &predictY)
{
    double sumx=0., sumy=0., sumxy=0., sumxx=0., sumyy=0.;
    for(vector<double>::size_type i=0; i<actualY.size(); ++i) {
        sumx  += actualY[i];
        sumy  += (predictY[i].y);
        sumxy += actualY[i]*(predictY[i].y);
        sumxx += actualY[i]*actualY[i];
        sumyy += (predictY[i].y)*(predictY[i].y);
    }
    int n = static_cast<int>(actualY.size());
    double meanx = sumx/n;
    double meany = sumy/n;
    double r = (sumxy-n*meanx*meany) / sqrt((sumxx-n*meanx*meanx)*(sumyy-n*meany*meany));
    return r;
}

vector<double> calcIAP(const vector<double> &actualY, const vector<PredictResult> &predictY)
{
    vector<double> iap;
    for(vector<PredictResult>::size_type i=0; i<predictY.size(); ++i) {
        if(predictY[i].som.empty())
            continue;
        int num_pos=0, num_neg=0, num_pos_neg=0;
        for(vector<double>::size_type j=0; j<predictY[i].som.size(); ++j) {
            if(!predictY[i].som[j])
                continue;
            ++num_pos;
            for(vector<double>::size_type k=0; k<predictY[i].som.size(); ++k) {
                if(predictY[i].som[k])
                    continue;
                ++num_neg;
                if(predictY[i].each_y[j] >= predictY[i].each_y[k])
                    ++num_pos_neg;
            }
        }
        if(num_pos && num_neg)
            iap.push_back(static_cast<double>(num_pos_neg) / (num_pos * num_neg));
    }

    return iap;
}

