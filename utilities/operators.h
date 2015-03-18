/*=============================================================================
 * #     FileName: operators.h
 * #         Desc:
 * #       Author: jlpeng
 * #        Email: jlpeng1201@gmail.com
 * #     HomePage:
 * #      Created: 2014-09-20 10:17:12
 * #   LastChange: 2014-11-01 05:03:09
 * #      History:
 * =============================================================================*/
#ifndef  UTILITIES_OPERATORS_H
#define  UTILITIES_OPERATORS_H

#include <ga/GA1DArrayGenome.h>

void myInitializer(GAGenome &genome);
int myMutator(GAGenome &genome, float pmut);
int myCrossover(const GAGenome &dad, const GAGenome &mom, GAGenome *bro, GAGenome *sis);

#endif

