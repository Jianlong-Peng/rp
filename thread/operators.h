/*=============================================================================
#     FileName: operators.h
#         Desc: 
#       Author: jlpeng
#        Email: jlpeng1201@gmail.com
#     HomePage: 
#      Created: 2014-09-20 10:17:12
#   LastChange: 2014-11-01 05:03:09
#      History:
=============================================================================*/
#ifndef  OPERATORS_H
#define  OPERATORS_H

#include <ga/GA1DArrayGenome.h>

/*
void myInitializer(GAGenome &genome);
int myMutator(GAGenome &genome, float pmut);
int myCrossover(const GAGenome &dad, const GAGenome &mom, GAGenome *bro, GAGenome *sis);
*/
float myEvaluator(GAGenome &genome);

extern bool cv_detail;

/*
// 1/mrss
float obj_1(std::vector<double>&, std::vector<PredictResult>&);
//float obj_2(std::vector<double>&, std::vector<PredictResult>&, GA1DArrayGenome<float>&);
// 1/mrss + mauc
float obj_3(std::vector<double>&, std::vector<PredictResult>&);
//float obj_4(std::vector<double>&, std::vector<PredictResult>&, GA1DArrayGenome<float>&);
// 1/mrss + mIAP
float obj_5(std::vector<double>&, std::vector<PredictResult>&);
*/

#endif   /* ----- #ifndef OPERATORS_H  ----- */

