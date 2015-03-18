/*=============================================================================
 #     FileName: operators.cpp
 #         Desc:
 #       Author: jlpeng
 #        Email: jlpeng1201@gmail.com
 #     HomePage:
 #      Created: 2015-03-18 11:56:12
 #   LastChange: 2015-03-18 12:00:02
 #      History:
 =============================================================================*/

#include "tools.h"
#include "operators.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <ga/GA1DArrayGenome.h>
#include <ga/garandom.h>

using std::vector;
using std::min_element;
using std::cout;
using std::cerr;
using std::endl;

extern Sample train_set;
extern int operator_type;
extern int num_types;
extern vector<bool> is_som;

inline float get_random_c(int cmin=-8, int cmax=8)
{
    return STA_CAST(float, pow(2., GARandomInt(cmin, cmax)));
}

inline float get_random_g(int gmin=-8, int gmax=8)
{
    return STA_CAST(float, pow(2., GARandomInt(gmin, gmax)));
}
inline float get_random_p(int pmin=1, int pmax=5)
{
    return STA_CAST(float, 0.05*GARandomInt(pmin, pmax));
}

void myInitializer(GAGenome &genome)
{
    GA1DArrayGenome<float> &g = DYN_CAST(GA1DArrayGenome<float> &, genome);
    g.resize(GAGenome::ANY_SIZE);
    int k = 0;
    float temp;
    for(int i=0; i<train_set.num_samples(); ++i) {
        float sum = 0.;
        // randomly initialization
        for(int j=0; j<train_set[i].num_atoms; ++j) {
            if(operator_type == 2) {
                if(is_som[k+j])
                    temp = GARandomFloat(0.5, 1.);
                else
                    temp =GARandomFloat(0., 0.5);
            }
            else
                temp = GARandomFloat(0., 1.);
            sum += temp;
            g.gene(k+j, temp);
        }
        // make sure summation over all atoms of molecule i is equal to be 1
        for(int j=0; j<train_set[i].num_atoms; ++j) {
            temp = g.gene(k+j);
            g.gene(k+j, temp/sum);
        }
        k += train_set[i].num_atoms;
    }
    // initialize c,g,p
    for(int i=0; i<num_types; ++i) {
        g.gene(k, get_random_c());
        g.gene(k+1, get_random_g());
        g.gene(k+2, get_random_p());
        k += 3;
    }
}

inline float generate_delta(float v1, float v2)
{
    float v[4] = {v1,v2,1-v1,1-v2};
    float minv = *min_element(v,v+4);
    return GARandomFloat(0., minv);
}

//swap mutator
// 1.
//   within each gene block of molecule i, randomly swap atom site i and j
//   such that summation over all atom sites of molecule i is always equal to 1!
// 2.
//   delta = min{vi, vj, 1-vi, 1-vj}
// 3.
//   if operator_type==1 or both i and j are SOM or non-SOM
//      gene(i)+/-delta, gene(i)+/-delta
//   otherwise
//      SOM site has more probability (see 0.7) to be +delta
//
int myMutator(GAGenome &genome, float pmut)
{
    GA1DArrayGenome<float> &g = DYN_CAST(GA1DArrayGenome<float>&, genome);
    if(pmut <= 0.)
        return 0;

    float p;
    float nMut = pmut * STA_CAST(float, g.length());
    if(nMut < 1.0) {
        nMut = 0;
        int k=0;
        for(int i=0; i<train_set.num_samples(); ++i) {
            for(int j=0; j<train_set[i].num_atoms; ++j) {
                if(GAFlipCoin(pmut)) {
                    int z = GARandomInt(0, train_set[i].num_atoms-1);
                    float delta = generate_delta(g.gene(k+j), g.gene(k+z));
                    if(operator_type==1 || (is_som[k+j]==is_som[k+z]))
                        p = 0.5;
                    else {
                        if(is_som[k+j])
                            p = 0.7;
                        else
                            p = 0.3;
                    }
                    if(GAFlipCoin(p)) {
                        g.gene(k+j, g.gene(k+j)+delta);
                        g.gene(k+z, g.gene(k+z)-delta);
                    } else {
                        g.gene(k+j, g.gene(k+j)-delta);
                        g.gene(k+z, g.gene(k+z)+delta);
                    }
                    //g.swap(k+j, GARandomInt(k, k+train_set[i].num_atoms-1));
                    ++nMut;
                }
            }
            k += train_set[i].num_atoms;
        }
    }
    else {
        for(int n=0; n<nMut; ++n) {
            int i = GARandomInt(0, train_set.num_samples()-1);
            int start = (i==0)?i:(train_set.count_num_atoms_until(i-1)-1);
            int end = start+train_set[i].num_atoms-1;
            int j = GARandomInt(start, end);
            int k = GARandomInt(start, end);
            float delta = generate_delta(g.gene(j), g.gene(k));
            if(operator_type==1 || (is_som[j]==is_som[k]))
                p = 0.5;
            else {
                if(is_som[j])
                    p = 0.7;
                else
                    p = 0.3;
            }
            if(GAFlipCoin(p)) {
                g.gene(j, g.gene(j)+delta);
                g.gene(k, g.gene(k)-delta);
            } else {
                g.gene(j, g.gene(j)-delta);
                g.gene(k, g.gene(k)+delta);
            }
            //g.swap(GARandomInt(start, end), GARandomInt(start, end));
        }
    }
    // mutation for c,g,p
    int k = train_set.count_total_num_atoms();
    for(int i=0; i<num_types; ++i) {
        if(GAFlipCoin(0.5))
            g.gene(k, get_random_c());
        if(GAFlipCoin(0.5))
            g.gene(k+1, get_random_g());
        if(GAFlipCoin(0.5))
            g.gene(k+2, get_random_p());
        k += 3;
    }

    return (STA_CAST(int,nMut));
}

// uniform crossover
// gene block of molecule i is either from mom or dad
int myCrossover(const GAGenome &p1, const GAGenome &p2, GAGenome *c1, GAGenome *c2)
{
    const GA1DArrayGenome<float> &mom = DYN_CAST(const GA1DArrayGenome<float>&, p1);
    const GA1DArrayGenome<float> &dad = DYN_CAST(const GA1DArrayGenome<float>&, p2);

    int n = 0;
    if(c1 && c2) {
        GA1DArrayGenome<float> &sis = DYN_CAST(GA1DArrayGenome<float>&, *c1);
        GA1DArrayGenome<float> &bro = DYN_CAST(GA1DArrayGenome<float>&, *c2);
        if(sis.length()!=bro.length() ||
                mom.length()!=dad.length() ||
                sis.length()!=mom.length()) {
            cerr << "Error: different gene length" << endl;
            exit(EXIT_FAILURE);
        }
        int k = 0;
        for(int i=0; i<train_set.num_samples(); ++i) {
            if(GAFlipCoin(0.5)) {
                for(int j=0; j<train_set[i].num_atoms; ++j) {
                    sis.gene(k+j, mom.gene(k+j));
                    bro.gene(k+j, dad.gene(k+j));
                }
            }
            else {
                for(int j=0; j<train_set[i].num_atoms; ++j) {
                    sis.gene(k+j, dad.gene(k+j));
                    bro.gene(k+j, mom.gene(k+j));
                }
            }
            k += train_set[i].num_atoms;
        }
        n = 2;

        // cross-over for c,g,p
        for(int i=0; i<num_types*3; ++i) {
            if(GAFlipCoin(0.5)) {
                sis.gene(k+i, mom.gene(k+i));
                bro.gene(k+i, dad.gene(k+i));
            }
            else {
                sis.gene(k+i, dad.gene(k+i));
                bro.gene(k+i, mom.gene(k+i));
            }
        }
    }
    else if(c1 || c2) {
        GA1DArrayGenome<float> &sis = (c1 ?
                DYN_CAST(GA1DArrayGenome<float>&, *c1):
                DYN_CAST(GA1DArrayGenome<float>&, *c2));
        if(sis.length()!=mom.length() ||
                mom.length()!=dad.length()) {
            cerr << "Error: different gene length" << endl;
            exit(EXIT_FAILURE);
        }
        int k = 0;
        for(int i=0; i<train_set.num_samples(); ++i) {
            if(GAFlipCoin(0.5)) {
                for(int j=0; j<train_set[i].num_atoms; ++j)
                    sis.gene(k+j, mom.gene(k+j));
            }
            else {
                for(int j=0; j<train_set[i].num_atoms; ++j)
                    sis.gene(k+j, dad.gene(k+j));
            }
            k += train_set[i].num_atoms;
        }
        n = 1;

        // crossover for c,g,p
        for(int i=0; i<num_types*3; ++i) {
            if(GAFlipCoin(0.5))
                sis.gene(k+i, mom.gene(k+i));
            else
                sis.gene(k+i, dad.gene(k+i));
        }
    }

    return n;
}



