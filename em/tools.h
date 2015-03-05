/*=============================================================================
#     FileName: tools.h
#         Desc: 
#       Author: jlpeng
#        Email: jlpeng1201@gmail.com
#     HomePage: 
#      Created: 2015-03-05 15:18:35
#   LastChange: 2015-03-05 20:26:48
#      History:
=============================================================================*/
#ifndef  EM_TOOLS_H
#define  EM_TOOLS_H

#include <vector>
#include "../partition/tools.h"
#include "../partition/svm.h"

class EM
{
public:
    explicit EM(Sample &sample);
    void init(bool test_som);
    // return: vector of deltas
    std::vector<double> run(int epochs, double epsilon,
            std::vector<double> expectation(const Sample&, std::vector<PredictResult>&));
    vector<double> &get_fraction() {return _fraction;}
    const vector<double> &get_fraction() const {return _fraction;}
    ~EM();
private:
    void fill_ys();
private:
    Sample &_sample;
    std::vector<svm_problem*> _probs;
    std::vector<double> _fraction;
};


#endif   /* ----- #ifndef EM_TOOLS_H  ----- */

