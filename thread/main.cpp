/*=============================================================================
#     FileName: main.cpp
#         Desc: 
#       Author: jlpeng
#        Email: jlpeng1201@gmail.com
#     HomePage: 
#      Created: 2014-09-15 09:13:56
#   LastChange: 2015-03-03 14:58:16
#      History:
=============================================================================*/
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iterator>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include "tools.h"
#include "../svm/svm.h"
#include "operators.h"
#include "extern_tools.h"
#include <ga/GA1DArrayGenome.h>
#include <ga/GASStateGA.h>

using namespace std;

// variable for GA
string ga_parameter_file("gapara.txt");
string log_init_file("log_init.txt");
string log_pop_file("log_pop.txt");
string log_stat_file("log_stat.txt");
unsigned int actual_seed(0);
int freq_flush(50);

// variable for input and output
string train_des_file("");
string train_som_file("");
string output_file("");

// extern variables
Sample train_set;
vector<int> num_xs;
vector<vector<svm_problem*> > probs;
int nfolds(5);
vector<vector<int> > perm;
vector<svm_parameter*> para;
int operator_type(1);
vector<bool> is_som;
int repeat(1);   // do cross-validation {repeat} times
int nthread(1);
//float (*obj_func)(vector<double>&, vector<PredictResult>&) = obj_1;
bool calc_auc(false);
bool calc_iap(false);
bool calc_consistency(false);
bool calc_x2(false);
double belta(1);
bool do_log(true);

int kernel_type(0);
vector<int> num_each_sample;
int num_types(0);
vector<vector<vector<double> > > kernel_matrix;
//vector<vector<int> > sample_atom_index;  // index of each atom according to its type



void parse_args(const char *infile);
//void randomize_samples();
//void construct_svm_problems_parameters();
//void free_svm_problems_parameters();
void output_best_genome(const GAGenome &genome, ostream &os);
void print_null(const char *s) {}

inline void print_time(time_t& start_time, std::ostream& out) {
    time_t now_time=time(NULL);
    out << "begin: " << ctime(&start_time);
    out << "now  : " << ctime(&now_time);
    int runtime = (int) difftime(now_time,start_time);
    out << "Total: " << runtime/3600 << " h, " << (runtime%3600)/60 << " m, " << (runtime%3600)%60 << " s" << std::endl;
}


int main(int argc, char *argv[])
{
    if(argc != 2) {
        cerr << endl << "  Usage: " << argv[0] << " para.txt" << endl
            << endl << "  obj function for GA:" << endl
            << "  1. \\frac{1}{\\mean{RSS}}" << endl
            << "  2. \\frac{1}{\\mean{RSS}} + \\mean{\\sum{xij^2}}" << endl
            << endl;
        exit(EXIT_FAILURE);
    }

    svm_set_print_string_function(print_null);

    parse_args(argv[1]);

    train_set.read_problem(train_des_file, train_som_file);
    cout << "size of training set: " << train_set.num_samples() << endl
        << "total number of atoms: " << train_set.count_total_num_atoms() << endl;

    if(train_som_file != "") {
        int k = 0;
        is_som.resize(train_set.count_total_num_atoms(), false);
        for(int i=0; i<train_set.num_samples(); ++i) {
            for(int j=0; j<train_set[i].num_atoms; ++j) {
                if(!train_set[i].som.empty() && train_set[i].som[j]==1)
                    is_som[k] = true;
                ++k;
            }
        }
    }

    // randomize samples for P times!!!!
    //GARandomSeed(actual_seed);
    //cout << "GARandomSeed()=" << GAGetRandomSeed() << endl;
    randomize_samples(true);
    construct_svm_problems_parameters();

    // atoms of all molecules followed by c,g,p
    GA1DArrayGenome<float> genome(train_set.count_total_num_atoms()+num_types*3);
    genome.initializer(myInitializer);
    genome.mutator(myMutator);
    genome.crossover(myCrossover);
    genome.evaluator(myEvaluator);

    time_t start_time = time(NULL);

    GAGeneticAlgorithm *ga = new GASteadyStateGA(genome);
    ga->parameters(ga_parameter_file.c_str());
    ga->initialize(actual_seed);
    
    ofstream log_init(log_init_file.c_str());
    log_init << "seed: " << GAGetRandomSeed() << endl << endl
        << "parameters: " << endl
        << ga->parameters() << endl;
    log_init.close();

    ofstream log_pop(log_pop_file.c_str());
    int count = 0;
    log_pop << "run: " << count << endl
        << ga->population();
    while(!ga->done()) {
        ++count;
        cout << "to do run: " << count << endl;
        ga->step();
        if(count%freq_flush == 0) {
            log_pop << "run: " << count << endl
                << ga->population();
            print_time(start_time, log_pop);

            ofstream log_stat(log_stat_file.c_str());
            log_stat << "run: " << count << endl
                << ga->statistics();
        }
    }
    ga->flushScores();

    log_pop << endl << endl << "-------" << endl;
    log_pop << "The GA found best:\n" << ga->statistics().bestIndividual() << endl;
    log_pop << "score: " << ga->statistics().bestIndividual().score() << endl;
    log_pop.close();

    // output best assignment
    // each line should be:
    // mol_name atom_id:CLi ...
    ofstream outf(output_file.c_str());
    if(!outf) {
        cerr << "Warning: failed to open " << output_file << endl
            << "         best assignments will be displayed in the screen" << endl;
        output_best_genome(ga->statistics().bestIndividual(), cout);
    }
    else {
        output_best_genome(ga->statistics().bestIndividual(), outf);
        outf.close();
    }

    free_svm_problems_parameters();

    return 0;
}

void parse_args(const char *infile)
{
    ifstream inf(infile);
    if(!inf) {
        cerr << "Error: failed to open " << infile << endl;
        exit(EXIT_FAILURE);
    }
    string line;
    //int obj_type(1);
    string obj_type("000");
    while(getline(inf,line)) {
        if(line.size()==0 || line[0]=='#')
            continue;
        string para;
        string value;
        istringstream is(line);
        is >> para;
        if(para == "gapara_file")
            is >> ga_parameter_file;
        else if(para == "train_des")
            is >> train_des_file;
        else if(para == "train_som")
            is >> train_som_file;
        else if(para == "output")
            is >> output_file;
        else if(para == "kernel_type") {
            is >> kernel_type;
            if(kernel_type<0 || kernel_type>2) {
                cerr << "Error: kernel_type should be one of 0,1,2, but " << kernel_type << " is given" << endl;
                exit(EXIT_FAILURE);
            }
        }
        else if(para == "seed")
            is >> actual_seed;
        else if(para == "freq_flush")
            is >> freq_flush;
        else if(para == "obj_type") {
            is >> obj_type;
            if(obj_type[0] == '1')
                calc_auc = true;
            if(obj_type.size()>=2 && obj_type[1]=='1')
                calc_iap = true;
            if(obj_type.size()>=3 && obj_type[2]=='1')
                calc_consistency = true;
            if(obj_type.size()>=4 && obj_type[3]=='1')
                calc_x2 = true;
            /*
            switch(obj_type) {
                case 1: obj_func = obj_1; break;
                //case 2: obj_func = obj_2; break;
                case 3: obj_func = obj_3; break;
                //case 4: obj_func = obj_4; break;
                case 5: obj_func = obj_5; break;
                default: 
                        cerr << "Error: `obj_type` should be 1,3,5, but " << obj_type << " is given" << endl;
                        exit(EXIT_FAILURE);
            }
            */
        }
        else if(para == "belta")
            is >> belta;
        else if(para == "do_log") {
            int temp;
            is >> temp;
            if(temp == 0)
                do_log = false;
        }
        else if(para == "operator_type")
            is >> operator_type;
        else if(para == "repeat")
            is >> repeat;
        else if(para == "nthread")
            is >> nthread;
        else
            cerr << "Warning: invalid parameter " << para << " being ignored" << endl;
    }

    if(train_des_file == "") {
        cerr << "Error: `train_des` missed" << endl;
        exit(EXIT_FAILURE);
    }
    if(output_file == "") {
        cerr << "Error: `output` missed" << endl;
        exit(EXIT_FAILURE);
    }
    if(operator_type==2 && train_som_file=="") {
        cerr << "Error: `operator_type=2`, but `train_som` is not given" << endl;
        exit(EXIT_FAILURE);
    }
    if(operator_type!=1 && operator_type!=2) {
        cerr << "Error: `operator_type` can only be 1 or 2, but " << operator_type << " is given" << endl;
        exit(EXIT_FAILURE);
    }
    if(repeat==1 && nthread>1) {
        cerr << "Warning: nthread will be set to be 1 when repeat=1" << endl;
        nthread = 1;
    }
    cout << "train_des: " << train_des_file << endl
        << "train_som: " << train_som_file << endl
        << "output file: " << output_file << endl
        << "operator_type: " << operator_type << endl
        << "obj_type: " << obj_type << endl
        << "  calc_auc: " << (calc_auc?"TRUE":"FALSE") << endl
        << "  calc_iap: " << (calc_iap?"TRUE":"FALSE") << endl
        << "  calc_consistency: " << (calc_consistency?"TRUE":"FALSE") << endl
        << "  calc_x2: " << (calc_x2?"TRUE":"FALSE") << endl
        << "belta: " << belta << endl
        << "do_log: " << (do_log?"TRUE":"FALSE") << endl
        << "repeat: " << repeat << endl
        << "nthread: " << nthread << endl;
}

// each line should be:
// mol_name atom_id:CLi ...
void output_best_genome(const GAGenome &genome, ostream &os)
{
    const GA1DArrayGenome<float> &g = DYN_CAST(const GA1DArrayGenome<float>&, genome);
    os << g.gene(0);
    for(int k=1; k<g.length(); ++k)
        os << " " << g.gene(k);
    os << endl;
    /*
    int k = 0;
    for(int i=0; i<train_set.num_samples(); ++i) {
        os << train_set[i].name;
        for(int j=0; j<train_set[i].num_atoms; ++j)
            os << " " << train_set[i].atom_id[j] << ":" << (log10(g.gene(k++))+train_set[i].y);
        os << endl;
    }
    for(; k<g.length(); k+=3)
        os << "c=" << g.gene(k) << " g=" << g.gene(k+1) << " p=" << g.gene(k+2) << endl;
    */
}

