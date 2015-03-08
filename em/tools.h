/*=============================================================================
#     FileName: tools.h
#         Desc: 
#       Author: jlpeng
#        Email: jlpeng1201@gmail.com
#     HomePage: 
#      Created: 2015-03-05 15:18:35
#   LastChange: 2015-03-06 14:26:16
#      History:
=============================================================================*/
#ifndef  EM_TOOLS_H
#define  EM_TOOLS_H

#include <ctime>
#include <vector>
#include "../partition/tools.h"
#include "../svm/svm.h"

class EM
{
public:
    explicit EM(Sample &sample);
    // randomly fill `_fraction`
    // if `test_som` is true, then SOM sites will be given a relatively larger value.
    void init(bool test_som, unsigned int seed=time(NULL));
    // if verbose is true, `delta` and `contrib` of each iteration will be displayed.
    void run(int epochs, double epsilon, bool verbose,
            std::vector<double> expectation(const Sample&, std::vector<PredictResult>&));
    vector<double> &get_fraction() {return _fraction;}
    const vector<double> &get_fraction() const {return _fraction;}
    vector<double> &get_cgp() {return _cgp;}
    const vector<double> &get_cgp() const {return _cgp;}
    ~EM();
private:
    // assign `_fraction` to `_probs`
    void fill_ys();
    // parameters will be optimized using grid search with 5-fold CV
    std::vector<svm_model*> train_models();
private:
    Sample &_sample;
    std::vector<svm_problem*> _probs;
    std::vector<double> _fraction;   // atom contribution (percentage)
    std::vector<double> _cgp;        // C,gamma,p for svm models
};

std::vector<double> expectation_rescale(const Sample&, std::vector<PredictResult>&);

#endif   /* ----- #ifndef EM_TOOLS_H  ----- */

