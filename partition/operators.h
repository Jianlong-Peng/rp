/*=============================================================================
#     FileName: operators.h
#         Desc: 
#       Author: jlpeng
#        Email: jlpeng1201@gmail.com
#     HomePage: 
#      Created: 2014-09-20 10:17:12
#   LastChange: 2014-10-31 21:20:01
#      History:
=============================================================================*/
#ifndef  OPERATORS_H
#define  OPERATORS_H

#include <vector>
#include <ga/GA1DArrayGenome.h>
#include "tools.h"

void myInitializer(GAGenome &genome);
int myMutator(GAGenome &genome, float pmut);
int myCrossover(const GAGenome &dad, const GAGenome &mom, GAGenome *bro, GAGenome *sis);
float myEvaluator(GAGenome &genome);

float obj_1(std::vector<double>&, std::vector<PredictResult>&, GA1DArrayGenome<float>&);
float obj_2(std::vector<double>&, std::vector<PredictResult>&, GA1DArrayGenome<float>&);
float obj_3(std::vector<double>&, std::vector<PredictResult>&, GA1DArrayGenome<float>&);
float obj_4(std::vector<double>&, std::vector<PredictResult>&, GA1DArrayGenome<float>&);
float obj_5(std::vector<double>&, std::vector<PredictResult>&, GA1DArrayGenome<float>&);

#endif   /* ----- #ifndef OPERATORS_H  ----- */

