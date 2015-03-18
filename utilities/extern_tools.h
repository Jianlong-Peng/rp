/*=============================================================================
#     FileName: extern_tools.h
#         Desc:
#       Author: jlpeng
#        Email: jlpeng1201@gmail.com
#     HomePage:
#      Created: 2015-03-18 10:51:31
#   LastChange: 2015-03-18 10:53:10
#      History:
=============================================================================*/
#ifndef  UTILITIES_EXTERN_TOOLS_H
#define  UTILITIES_EXTERN_TOOLS_H

#include "tools.h"
#include <vector>

using std::vector;

/*
 * extern bool calc_auc;
 * extern bool calc_iap;
 * extern bool calc_consistency;
 * extern bool calc_x2;
 * extern Sample train_set;
 * extern bool belta;
 * extern bool wx2;
 */
float obj(vector<double> &actualY, vector<PredictResult> &predictY,
    vector<int> &sample_index, vector<float> &population, bool verbose=false);

#endif

