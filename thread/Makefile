GXX = icc
GA_ROOT = /home/xmluo/jlpeng/program/ga-svm/galib247
CFLAGS = -I$(GA_ROOT) -Wall -O2 -openmp
#CFLAGS = -I$(GA_ROOT) -Wall -g
LDFLAGS = -L$(GA_ROOT)/ga -lga

OBJS = ../utilities/tools.o ../svm/svm.o extern_tools.o

EXES = ga_partition gap_cv gap gap_best gap_analysis

all: $(EXES)

%.o: %.cpp
	@echo to build $@
	@$(GXX) $(CFLAGS) $< -c

ga_partition: $(OBJS) operators.o main.o ../utilities/extern_tools.o ../utilities/operators.o
	@echo to build ga_partition
	@$(GXX) -openmp ../utilities/operators.o ../utilities/extern_tools.o $(OBJS) operators.o main.o $(LDFLAGS) -o ga_partition

gap: ga_partition
	@echo to create gap
	@ln -s ga_partition gap

gap_cv: $(OBJS) gap_cv.o
	@echo to build gap_cv
	@$(GXX) -openmp $(OBJS) gap_cv.o $(LDFLAGS) -o gap_cv

gap_best: $(OBJS) operators.o gap_best.o ../utilities/extern_tools.o
	@echo to build gap_best
	@$(GXX) -openmp ../utilities/extern_tools.o $(OBJS) operators.o gap_best.o $(LDFLAGS) -o gap_best

gap_analysis: $(OBJS) operators.o gap_analysis.o ../utilities/extern_tools.o
	@echo to build gap_analysis
	@$(GXX) -openmp ../utilities/extern_tools.o $(OBJS) operators.o gap_analysis.o $(LDFLAGS) -o gap_analysis

clean:
	rm *.o $(EXES)

