GA_ROOT = /home/xmluo/jlpeng/program/ga-svm/galib247
CFLAGS = -I$(GA_ROOT) -Wall -O2
#CFLAGS = -I$(GA_ROOT) -Wall -g
CFLAGS_P = -I$(GA_ROOT) -Wall -O2 -DNTHREAD
#CFLAGS_P = -I$(GA_ROOT) -Wall -g -DNTHREAD
LDFLAGS = -L$(GA_ROOT)/ga -lga

OBJS = ../utilities/tools.o ../svm/svm.o

EXES = ga_partition gap_train gap_predict gap_train_p gap_cv write_svm_problem gap_analysis gap_random gap_sites gap

all: $(EXES)

gap_train_p.o: gap_train.cpp
	@echo to build gap_train_p.o
	@g++ $(CFLAGS_P) gap_train.cpp -c -o gap_train_p.o

%.o: %.cpp
	@echo to build $@
	@g++ $(CFLAGS) $< -c

ga_partition: $(OBJS) main.o extern_tools.o ../utilities/extern_tools.o ../utilities/operators.o
	@echo to build ga_partition
	@g++ $(OBJS) ../utilities/operators.o ../utilities/extern_tools.o extern_tools.o main.o $(LDFLAGS) -o ga_partition

gap: ga_partition
	@echo to create gap
	@ln -s ga_partition gap

gap_train: $(OBJS) ../svm/svmtools.o gap_train.o
	@echo to build gap_train
	@g++ $(OBJS) ../svm/svmtools.o gap_train.o -o gap_train

gap_train_p: $(OBJS) ../svm/svmtools.o gap_train_p.o
	@echo to build gap_train_p
	@g++ $(OBJS) ../svm/svmtools.o gap_train_p.o -o gap_train_p -lpthread

gap_predict: $(OBJS) gap_predict.o
	@echo to build gap_predict
	@g++ $(OBJS) gap_predict.o -o gap_predict

gap_cv: $(OBJS) gap_cv.o extern_tools.o
	@echo to build gap_cv
	@g++ $(OBJS) extern_tools.o gap_cv.o $(LDFLAGS) -o gap_cv

gap_analysis: $(OBJS) gap_analysis.o extern_tools.o
	@echo to build gap_analysis
	@g++ $(OBJS) extern_tools.o gap_analysis.o $(LDFLAGS) -o gap_analysis

gap_random: $(OBJS) gap_random.o extern_tools.o
	@echo to build gap_random
	@g++ $(OBJS) extern_tools.o gap_random.o $(LDFLAGS) -o gap_random

gap_sites: $(OBJS) gap_sites.o
	@echo t build gap_sites
	@g++ $(OBJS) gap_sites.o $(LDFLAGS) -o gap_sites

write_svm_problem: $(OBJS) write_svm_problem.o
	@echo to build write_svm_problem
	@g++ $(OBJS) write_svm_problem.o $(LDFLAGS) -o write_svm_problem

test: svm.o test.cpp
	g++ -Wall -O2 svm.o test.cpp -o test -lpthread

svm-train: svm-train.c svm.o
	g++ $(CFLAGS) svm.o svm-train.c -o svm-train

svm-predict: svm-predict.c svm.o
	g++ $(CFLAGS) svm.o svm-predict.c -o svm-predict

clean:
	rm *.o $(EXES)


