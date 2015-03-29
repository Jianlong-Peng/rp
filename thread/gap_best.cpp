/*=============================================================================
#     FileName: gap_best.cpp
#         Desc: 
#       Author: jlpeng
#        Email: jlpeng1201@gmail.com
#     HomePage: 
#      Version: 0.0.1
#   LastChange: 2015-03-07 05:42:42
#      History:
=============================================================================*/
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iterator>
#include "../utilities/tools.h"
#include "extern_tools.h"
#include "operators.h"
#include <omp.h>
#include <ga/GA1DArrayGenome.h>
#include <ga/garandom.h>


using namespace std;

string train_file("");
string out_file("");
string som_file("");
string pop_file("");
int run(-1);
vector<GA1DArrayGenome<float> > genomes;
unsigned actual_seed(0);
int freq_flush(50);

// extern variables
// the following is filled by corresponding functions
// randomize_samples & construct_svm_problems_parameter
int num_types;
Sample train_set;
vector<vector<svm_problem*> > probs;
vector<svm_parameter*> para;
vector<vector<int> > perm;
vector<int> num_each_sample;
vector<int> num_xs;
// the following is read from CMD option
bool calc_auc(false);
bool calc_iap(false);
bool calc_consistency(true);
bool calc_x2(false);
int nthread(5);
int nfolds(5);
int repeat(30);
double belta(1.);
double wx2(1.);
double wauc(1.);
// the following is fixed.
int kernel_type(0);      // RBF
int operator_type(1);    // not used here
vector<bool> is_som;     // not used here
bool do_log(true);
vector<vector<vector<double> > > kernel_matrix;  // not used here

void print_null(const char *s) {}
void exit_with_help(const char *name);
void parse_args(const char *infile);
void load_population();
void find_best_genome();

int main(int argc, char *argv[])
{
    cout << "CMD:";
    for(int i=0; i<argc; ++i)
        cout << " " << argv[i];
    cout << endl;

    //if(argc < 5)
    //    exit_with_help(argv[0]);
    if(argc != 5)
        exit_with_help(argv[0]);

    //parse_options(argc, argv);
    parse_args(argv[3]);
    pop_file = argv[1];
    run = atoi(argv[2]);
    out_file = argv[4];

    svm_set_print_string_function(print_null);

    train_set.read_problem(train_file, som_file);

    GARandomSeed(actual_seed);
    randomize_samples(true);
    construct_svm_problems_parameters();

    cout << "train_des: " << train_file << endl
        << "train_som: " << som_file << endl
        << "output file: " << out_file << endl
        << "number of samples: " << train_set.count_total_num_atoms() << endl
        << "probs.size(): " << probs.size() << endl
        << "para.size(): " << para.size() << endl
        << "perm.size(): " << perm.size() << endl
        << "num_types: " << num_types << endl
        << "num_each_sample: ";
    copy(num_each_sample.begin(),num_each_sample.end(),ostream_iterator<int>(cout," "));
    cout << endl
        << "num_xs: ";
    copy(num_xs.begin(), num_xs.end(), ostream_iterator<int>(cout," "));
    cout << endl
        << "  calc_auc: " << (calc_auc?"TRUE":"FALSE") << endl
        << "  calc_iap: " << (calc_iap?"TRUE":"FALSE") << endl
        << "  calc_consistency: " << (calc_consistency?"TRUE":"FALSE") << endl
        << "  calc_x2: " << (calc_x2?"TRUE":"FALSE") << endl
        << "nthread: " << nthread << endl
        << "nfolds: " << nfolds << endl
        << "repeat: " << repeat << endl
        << "belta: " << belta << endl
        << "wx2: " << wx2 << endl
        << "wauc: " << wauc << endl;

    load_population();
    cout << "genomes.size(): " << genomes.size() << endl;

    find_best_genome();


    return 0;
}

void exit_with_help(const char *name)
{
    cerr << endl << "OBJ" << endl
        << "  to extract the best genome from population" << endl
        << endl << "Usage" << endl
        << "  " << name << " pop_file run para.txt output" << endl
        << endl << "  pop_file: log_pop.txt" << endl
        << "  run     : int, specify which run of population to be analyzed" << endl
        << "  para.txt: same as input for `gap`" << endl
        << "  output  : where to save the best genome" << endl
        << endl;
    exit(EXIT_FAILURE);
}

void exit_with_help_2(const char *name)
{
    cerr << endl << "OBJ" << endl
        << "  to extract the best genome from population" << endl
        << "  each genome will be evaluated using {repeat} times of 5-fold CV" << endl
        << endl << "Usage" << endl
        << "  " << name << " [options]" << endl
        << endl << "[options]" << endl
        << "  --train   file:" << endl
        << "  --pop     file: log_pop.txt" << endl
        << "  --run      int: specify which run of population to be analyzed" << endl
        << "  --out     file: where to save the best genome" << endl
        << "  --som     file: <optional>" << endl
        << "  --obj      str: <default: 0010>" << endl
        << "    apart from RSS, the following values will be calculated" << endl
        << "    if the corresponding bit is set to be 1" << endl
        << "      1 - AUC  (--som is needed)" << endl
        << "      2 - IAP  (--som is needed)" << endl
        << "      3 - consistency" << endl
        << "      4 - mean{sum_j{x_ij^2}}" << endl
        << "    by default, comparsion is made based on (1/RSS+belta*1/consistency)" << endl
        << "  --repeat  int : <default: 30>" << endl
        << "    number of times to do 5-fold CV" << endl
        << "  --nfold   int : <default: 5>" << endl
        << "  --belta double: <default: 1.>" << endl
        << "  --np      int : <default: 5>" << endl
        << "  --seed    int : <default: 0>" << endl
        << endl;
    exit(EXIT_FAILURE);
}

void parse_args(const char *infile)
{
    ifstream inf(infile);
    if(!inf) {
        cerr << "Error: failed to open " << infile << endl;
        exit(EXIT_FAILURE);
    }
    string line;
    string obj_type("000");
    while(getline(inf,line)) {
        if(line.size()==0 || line[0]=='#')
            continue;
        string para;
        string value;
        istringstream is(line);
        is >> para;
        //if(para == "gapara_file")
        //    is >> ga_parameter_file;
        if(para == "train_des")
            is >> train_file;
        else if(para == "train_som")
            is >> som_file;
        //else if(para == "output")
        //    is >> out_file;
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
        }
        else if(para == "belta")
            is >> belta;
        else if(para == "wx2")
            is >> wx2;
        else if(para == "wauc")
            is >> wauc;
        else if(para == "do_log") {
            int temp;
            is >> temp;
            if(temp == 0)
                do_log = false;
        }
        //else if(para == "operator_type")
        //    is >> operator_type;
        else if(para == "repeat")
            is >> repeat;
        else if(para == "nthread")
            is >> nthread;
        else
            cerr << "Warning: invalid parameter " << para << " being ignored" << endl;
    }
    inf.close();

    if(train_file == "") {
        cerr << "Error: `train_des` missed" << endl;
        exit(EXIT_FAILURE);
    }
    if(repeat==1 && nthread>1) {
        cerr << "Warning: nthread will be set to be 1 when repeat=1" << endl;
        nthread = 1;
    }
}

void parse_options(int argc, char *argv[])
{
    int i;
    for(i=1; i<argc; ++i) {
        if(argv[i][0] != '-')
            break;
        if(strcmp(argv[i],"--train") == 0)
            train_file = argv[++i];
        else if(strcmp(argv[i],"--out") == 0)
            out_file = argv[++i];
        else if(strcmp(argv[i],"--som") == 0)
            som_file = argv[++i];
        else if(strcmp(argv[i],"--pop") == 0)
            pop_file = argv[++i];
        else if(strcmp(argv[i],"--run") == 0)
            run = atoi(argv[++i]);
        else if(strcmp(argv[i],"--nfold") == 0)
            nfolds = atoi(argv[++i]);
        else if(strcmp(argv[i], "--np") == 0)
            nthread = atoi(argv[++i]);
        else if(strcmp(argv[i], "--belta") == 0)
            belta = atof(argv[++i]);
        else if(strcmp(argv[i], "--seed") == 0) {
            istringstream is(argv[++i]);
            is >> actual_seed;
        }
        else if(strcmp(argv[i],"--obj") == 0) {
            char *val = argv[++i];
            if(strlen(val) != 3) {
                cerr << "Error: invalid `--obj " << val << "`" << endl;
                exit(EXIT_FAILURE);
            }
            if(val[0] == '1')
                calc_auc = true;
            if(val[1] == '1')
                calc_iap = true;
            if(val[2] == '0')
                calc_consistency = false;
        }
        else if(strcmp(argv[i],"--repeat") == 0)
            repeat = atoi(argv[++i]);
        else {
            cerr << "Error: invalid option " << argv[i] << endl;
            exit(EXIT_FAILURE);
        }
    }
    if(argc-i != 0) {
        cerr << "Error: invalid number of arguments" << endl;
        exit(EXIT_FAILURE);
    }
    if(train_file.empty()) {
        cerr << "Error: miss `--train file`" << endl;
        exit(EXIT_FAILURE);
    }
    if(pop_file.empty()) {
        cerr << "Error: miss `--pop file`" << endl;
        exit(EXIT_FAILURE);
    }
    if(run == -1) {
        cerr << "Error: miss `--run int`" << endl;
        exit(EXIT_FAILURE);
    }
}

void load_population()
{
    ifstream inf(pop_file.c_str());
    if(!inf) {
        cerr << "Error: failed to open " << pop_file << endl;
        exit(EXIT_FAILURE);
    }
    string line;
    int each_run;
    bool found(false);
    getline(inf,line);
    while(!inf.eof()) {
        istringstream is(line);
        string temp;
        is >> temp >> each_run;
        if(each_run != run) {
            getline(inf,line);
            while(!inf.eof() && line.find("run")!=0)
                getline(inf,line);
        }
        else {
            while(getline(inf,line)) {
                if(line.size()==0 || line.find("run")==0)
                    break;
                istringstream iss(line);
                float val;
                GA1DArrayGenome<float> genome(train_set.count_total_num_atoms()+num_types*3);
                for(int i=0; iss >> val; ++i)
                    genome.gene(i, val);
                genomes.push_back(genome);
            }
            found = true;
            break;
        }
    }
    inf.close();
    if(!found) {
        cerr << "Error: can't find run=" << run << " from " << pop_file << endl;
        exit(EXIT_FAILURE);
    }
}

void find_best_genome()
{
    float best_val(-1E8);
    vector<GA1DArrayGenome<float> >::size_type best_i(genomes.size());

    for(vector<GA1DArrayGenome<float> >::size_type i=0; i<genomes.size(); ++i) {
        float val = myEvaluator(genomes[i]);
        cout << "genome " << i+1 << " OBJ=" << val << endl;
        if(val > best_val) {
            best_val = val;
            best_i = i;
        }
    }
    cout << "=> best genome " << best_i+1 << " val=" << best_val << endl;

    ofstream outf(out_file.c_str());
    if(!outf) {
        cerr << "Error: failed to open " << out_file << endl;
        cout << genomes[best_i].gene(0);
        for(int j=0; j<genomes[best_i].length(); ++j)
            cout << " " << genomes[best_i].gene(j);
        cout << endl;
    }
    else {
        outf << genomes[best_i].gene(0);
        for(int j=1; j<genomes[best_i].length(); ++j)
            outf << " " << genomes[best_i].gene(j);
        outf << endl;
        outf.close();
    }
}


