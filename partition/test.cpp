/*=============================================================================
#     FileName: test.cpp
#         Desc: 
#       Author: jlpeng
#        Email: jlpeng1201@gmail.com
#     HomePage: 
#      Created: 2014-09-18 22:08:13
#   LastChange: 2014-09-19 05:16:11
#      History:
=============================================================================*/
#include <iostream>
#include <queue>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cerrno>
#include <pthread.h>
#include "svm.h"

using namespace std;

struct CPG
{
    CPG(int i, int j, int k): c(pow(2,i)), g(pow(2,j)), p(0.5*k) {}

    double c;
    double g;
    double p;
};

pthread_mutex_t mut;
double best_rmse(1e38);
double best_c(1.);
double best_g(1.);
double best_p(1.);
svm_problem *prob(NULL);
queue<CPG> candidates;


void read_svm_problem(const char *filename);
void grid_search_p(int num_thread);
void grid_search();
void free_svm_problem();
void print_null(const char *s) {}

int main(int argc, char *argv[])
{
    if(argc!=2 && argc!=3) {
        cerr << endl << "  Usage: " << argv[0] << " in.svm [num_threads]" << endl
            << "  [num_threads] should be >= 0" << endl
            << "  if >0, then do grid search using [num_threads] threads" << endl
            << endl;
        exit(EXIT_FAILURE);
    }

    int num_thread = 0;

    if(argc == 3) {
        num_thread = atoi(argv[2]);
        if(num_thread < 0) {
            cerr << "Error: num_thread should be positive, but given " << num_thread << endl;
            exit(EXIT_FAILURE);
        }
    }

    svm_set_print_string_function(print_null);

    read_svm_problem(argv[1]);

    for(int i=0; i<1; ++i) {
        cout << "========================" << endl
            << "# " << i+1 << "th iteration" << endl
            << "========================" << endl;
        if(num_thread)
            grid_search_p(num_thread);
        else
            grid_search();
        cout << "best: c=" << best_c << ", g=" << best_g << ", p=" << best_p << ", rmse=" << best_rmse << endl;
    }

    free_svm_problem();

    return 0;
}

void free_svm_problem()
{
    if(prob == NULL)
        return;
    if(prob->y)
        free(prob->y);
    if(prob->x) {
        free(prob->x[0]);
        free(prob->x);
    }
    free(prob);
}

static char* readline(FILE *input)
{
    int len;
    int max_line_len = 1024;
    char *line = (char*)malloc(sizeof(char)*max_line_len);
    
    if(fgets(line,max_line_len,input) == NULL)
        return NULL;

    while(strrchr(line,'\n') == NULL)
    {
        max_line_len *= 2;
        line = (char *) realloc(line,max_line_len);
        len = (int) strlen(line);
        if(fgets(line+len,max_line_len-len,input) == NULL)
            break;
    }
    return line;
}
static void exit_input_error(int line_num)
{
    fprintf(stderr,"Wrong input format at line %d\n", line_num);
    exit(1);
}
void read_svm_problem(const char *filename)
{
    int elements, max_index, inst_max_index, i, j;
    FILE *fp;
    char *endptr;
    char *idx, *val, *label;
    char *line;
    char *p;
    struct svm_node *x_space;

    fp = fopen(filename, "r");
    if(fp == NULL)
    {
        fprintf(stderr,"can't open input file %s\n",filename);
        exit(1);
    }

    prob = (svm_problem*)malloc(sizeof(svm_problem));
    prob->l = 0;
    elements = 0;

    while((line = readline(fp))!=NULL)
    {
        p = strtok(line," \t"); // label

        // features
        while(1)
        {
            p = strtok(NULL," \t");
            if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
                break;
            ++elements;
        }
        ++elements;
        ++prob->l;
    }
    rewind(fp);

    prob->y = (double*)malloc(sizeof(double)*(prob->l));
    prob->x = (svm_node**)malloc(sizeof(svm_node*)*(prob->l));
    x_space = (svm_node*)malloc(sizeof(svm_node)*elements);

    max_index = 0;
    j=0;
    for(i=0;i<prob->l;i++)
    {
        inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
        line = readline(fp);
        prob->x[i] = &x_space[j];
        label = strtok(line," \t");
        prob->y[i] = strtod(label,&endptr);
        if(endptr == label)
            exit_input_error(i+1);

        while(1)
        {
            idx = strtok(NULL,":");
            val = strtok(NULL," \t");

            if(val == NULL)
                break;

            errno = 0;
            x_space[j].index = (int) strtol(idx,&endptr,10);
            if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
                exit_input_error(i+1);
            else
                inst_max_index = x_space[j].index;

            errno = 0;
            x_space[j].value = strtod(val,&endptr);
            if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
                exit_input_error(i+1);

            ++j;
        }

        if(inst_max_index > max_index)
            max_index = inst_max_index;
        x_space[j++].index = -1;
    }

    fclose(fp);
}

static double calcRMSE(double *target)
{
    double rmse = 0.;
    for(int i=0; i<prob->l; ++i)
        rmse += pow(prob->y[i]-target[i], 2);
    return sqrt(rmse / prob->l);
}

void do_cv_and_update(double c, double g, double p)
{
    // create parameters
    svm_parameter *para = (svm_parameter*)malloc(sizeof(svm_parameter));
    para->svm_type = EPSILON_SVR;
    para->kernel_type = RBF;
    para->degree = 3;
    para->gamma = g;
    para->coef0 = 0;
    para->cache_size = 100;
    para->eps = 0.001;
    para->C = c;
    para->nr_weight = 0;
    para->weight_label = NULL;
    para->weight = NULL;
    para->nu = 0.5;
    para->p = p;
    para->shrinking = 1;
    para->probability = 0;

    double *target = (double*)malloc(sizeof(double)*(prob->l));
    svm_cross_validation(prob, para, 5, target);
    double rmse = calcRMSE(target);
    pthread_mutex_lock(&mut);
    cout << "c=" << c << ", g=" << g << ", p=" << p << ", rmse=" << rmse << endl;
    if(rmse < best_rmse) {
        best_c = c;
        best_g = g;
        best_p = p;
        best_rmse = rmse;
    }
    pthread_mutex_unlock(&mut);

    free(target);
    free(para);
}

void *thread_cv(void *arg)
{
    double c,p,g;
    while(1) {
        bool end = false;
        pthread_mutex_lock(&mut);
        //cout << (char*)arg << endl;
		cout << "thread " << *(int*)arg << endl;
        if(candidates.empty()) {
            end = true;
        }
        else {
            CPG temp = candidates.front();
            c = temp.c;
            p = temp.p;
            g = temp.g;
            candidates.pop();
        }
        pthread_mutex_unlock(&mut);
        if(end)
            break;
        else
            do_cv_and_update(c,g,p);        
    }
    pthread_exit(arg);
}

void grid_search_p(int num_thread)
{
    for(int i=1; i<=5; ++i)
        for(int j=-8; j<=8; ++j)
            for(int k=-8; k<=8; ++k)
                candidates.push(CPG(k,j,i));
    char *thread_names[] = {"thread 1", "thread 2", "thread 3", "thread 4", 
                "thread 5", "thread 6", "thread 7", "thread 8",
                "thread 9", "thread 10"};
	int thread_name[] = {1,2,3,4,5,6,7,8,9,10};

    pthread_t *thread = (pthread_t*)malloc(sizeof(pthread_t)*num_thread);
    memset(thread, 0, sizeof(pthread_t)*num_thread);
    for(int i=0; i<num_thread; ++i) {
		//int val = pthread_create(&thread[i], NULL, thread_cv, thread_names[i]);
        int val = pthread_create(&thread[i], NULL, thread_cv, &thread_name[i]);
        if(val)
            cerr << "Error: failed to create thread " << i+1 << endl;
    }
    for(int i=0; i<num_thread; ++i) {
		//char *temp;
        //int val = pthread_join(thread[i], (void**)(&temp));
		int *temp;
		int val = pthread_join(thread[i], (void**)(&temp));
		cerr << *temp << " done" << endl;
        if(val != 0)
            cerr << "Error: failed to join thread " << i+1 << "; val=" << val << endl;
    }

    free(thread);

}

void grid_search()
{
    svm_parameter *para = (svm_parameter*)malloc(sizeof(svm_parameter));
    para->svm_type = EPSILON_SVR;
    para->kernel_type = RBF;
    para->degree = 3;
    para->gamma = 1.;
    para->coef0 = 0;
    para->cache_size = 100;
    para->eps = 0.001;
    para->C = 1.;
    para->nr_weight = 0;
    para->weight_label = NULL;
    para->weight = NULL;
    para->nu = 0.5;
    para->p = 1.;
    para->shrinking = 1;
    para->probability = 0;

    double *target = (double*)malloc(sizeof(double)*(prob->l));

    for(int i=1; i<=5; ++i) {
        para->p = 0.5*i;
        for(int j=-8; j<=8; ++j) {
            para->gamma = pow(2, j);
            for(int k=-8; k<=8; ++k) {
                para->C = pow(2,k);
                svm_cross_validation(prob, para, 5, target);
                double rmse = calcRMSE(target);
                cout << "c=" << para->C << ", g=" << para->gamma << ", p=" << para->p << ", rmse=" << rmse << endl;
                if(rmse < best_rmse) {
                    best_c = para->C;
                    best_g = para->gamma;
                    best_p = para->p;
                    best_rmse = rmse;
                }
            }
        }
    }

    free(para);
    free(target);
}

