GA_ROOT = /home/xmluo/jlpeng/program/ga-svm/galib247
CFLAGS = -I$(GA_ROOT) -Wall -O2
#CFLAGS = -I$(GA_ROOT) -Wall -g
LDFLAGS = -L$(GA_ROOT)/ga -lga -lpthread

OBJS = tools.o ../svm/svm.o extern_tools.o

EXES = ga_partition gap_cv gap gap_best

all: $(EXES)

%.o: %.cpp
	@echo to build $@
	@g++ $(CFLAGS) $< -c

ga_partition: $(OBJS) operators.o main.o
	@echo to build ga_partition
	@g++ $(OBJS) operators.o main.o $(LDFLAGS) -o ga_partition

gap: ga_partition
	ln -s ga_partition gap

gap_cv: $(OBJS) gap_cv.o
	@echo to build gap_cv
	@g++ $(OBJS) gap_cv.o $(LDFLAGS) -o gap_cv

gap_best: $(OBJS) operators.o gap_best.o
	@echo to build gap_best
	@g++ $(OBJS) operators.o gap_best.o $(LDFLAGS) -o gap_best

clean:
	rm *.o $(EXES)

