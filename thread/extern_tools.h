/*=============================================================================
#     FileName: extern_tools.h
#         Desc: 
#       Author: jlpeng
#        Email: jlpeng1201@gmail.com
#     HomePage: 
#      Created: 2014-10-27 14:05:32
#   LastChange: 2014-10-27 14:21:33
#      History:
=============================================================================*/
#ifndef  EXTERN_TOOLS_H
#define  EXTERN_TOOLS_H

#include <vector>
#include "../utilities/tools.h"

using std::vector;

// vector<int> perm
// if verbose=true, display randomized samples (indices)
void randomize_samples(bool verbose=false);
// int num_types; vector<int> num_xs; vector<int> num_each_sample;
// vector<svm_problem*> probs; vector<vector<vector<double> > > kernel_matrix;
// svm_parameter *para;
void construct_svm_problems_parameters();
// free probs, para
void free_svm_problems_parameters();

#endif   /* ----- #ifndef EXTERN_TOOLS_H  ----- */

