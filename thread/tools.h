/*=============================================================================
#     FileName: tools.h
#         Desc: 
#       Author: jlpeng
#        Email: jlpeng1201@gmail.com
#     HomePage: 
#      Created: 2014-09-15 19:50:31
#   LastChange: 2014-10-31 15:57:25
#      History:
=============================================================================*/

#ifndef  GACL_TOOLS_H
#define  GACL_TOOLS_H

#include <string>
#include <vector>
#include <algorithm>
#include "svm.h"

using std::vector;

struct PredictResult
{
    PredictResult(): y(0.) {each_y.clear(); som.clear();}
    // all are log10() transformed
    // so, y = log10(sum{10^each_y})
    double y;
    vector<double> each_y;
    vector<bool> som;  // len(som) == len(each_y); som[i]=true/false; Attention: maybe empty!!!!
};

class Sample;
struct Molecule
{
    Molecule(): hasy(false), y(1e38), num_atoms(0) {}
    PredictResult predict(vector<svm_model*> &models);
    PredictResult predict(vector<svm_model*> &models, Sample &train, 
        double (*calcKernel)(vector<double> &x, vector<double> &y));
    std::string name;
    bool hasy;
    double y;   // log10(CLint)
    int num_atoms;
    vector<vector<double> > x;
    vector<double> each_y;    // empty or (predicted) atom contribution; log10 transformed;
    vector<bool> som;   // if not empty, then same length as atom_id, and each will be true or false
    vector<int> atom_id;
    vector<int> atom_type;
};

class Sample
{
public:
    void read_problem(std::string train_des_file, std::string train_som_file="");
    // same content as input file `train_des_file`
    void write_problem(std::string outfile);
    // write svm problems of all atom types
    void write_svm_problem(std::string outfile);
    // write svm problem of atom type `type`
    void write_svm_problem(std::string outfile, int type);
    inline Molecule &operator[](int i) {return data[i];}
    inline const Molecule &operator[](int i) const {return data[i];}
    inline int num_samples() const {return static_cast<int>(data.size());}
    int count_total_num_atoms() const {
        return count_num_atoms_until(data.size());
    }
    // count number of atoms from molecules 0 to `i`
    int count_num_atoms_until(unsigned int i) const  {
        int count = 0;
        if(i >= data.size()) i = data.size()-1;
        for(unsigned int j=0; j<=i; ++j)
            count += (data[j].num_atoms);
        return count;
    }
    int number_atoms_of_type(int type);
    inline bool has_som() const {return som;}
    vector<PredictResult> predict(vector<svm_model*> &models);
    vector<PredictResult> predict(vector<svm_model*> &models, Sample &train, 
        double (*calcKernel)(vector<double> &x, vector<double> &y));

private:
    vector<Molecule> data;
    bool som;
};

// labels[i]=true means it's positive; otherwise negative
double calcAUC(const vector<bool> &labels, const vector<double> &ys);
double calcRSS(const vector<double> &actualY, const vector<double> &predictY);
double calcRSS(const vector<double> &actualY, const vector<PredictResult> &predictY);
double calcRMSE(const vector<double> &actualY, const vector<double> &predictY);
double calcRMSE(const vector<double> &actualY, const vector<PredictResult> &predictY);
double calcR(const vector<double> &actualY, const vector<double> &predictY);
double calcR(const vector<double> &actualY, const vector<PredictResult> &predictY);
vector<double> calcIAP(const vector<double> &actualY, const vector<PredictResult> &predictY);

template<class type>
double tanimotoKernel(vector<type> &X, vector<type> &Y)
{
    type xy(0), xx(0), yy(0);
    for(unsigned i=0; i<X.size(); ++i) {
        xy += X[i]*Y[i];
        xx += X[i]*X[i];
        yy += Y[i]*Y[i];
    }
    return (static_cast<double>(xy) / (xx+yy-xy));
}

template<class type>
double minMaxKernel(vector<type> &X, vector<type> &Y)
{
    type numerator(0), denominator(0);
    for(unsigned i=0; i<X.size(); ++i) {
        numerator += std::min(X[i], Y[i]);
        denominator += std::max(X[i], Y[i]);
    }
    return (static_cast<double>(numerator) / denominator);
}

#endif   /* ----- #ifndef GACL_TOOLS_H  ----- */

