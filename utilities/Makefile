GA_ROOT = /home/xmluo/jlpeng/program/ga-svm/galib247
CFLAGS = -I$(GA_ROOT) -Wall -O2
#CFLAGS = -I$(GA_ROOT) -Wall -g
CFLAGS_P = -I$(GA_ROOT) -Wall -O2 -DNTHREAD
#CFLAGS_P = -I$(GA_ROOT) -Wall -g -DNTHREAD
#LDFLAGS = -L$(GA_ROOT)/ga -lga

OBJS = tools.o extern_tools.o operators.o

all: $(OBJS) test_auc

%.o: %.cpp
	@echo to build $@
	@g++ $(CFLAGS) $< -c

test_auc: test_auc.o tools.o ../svm/svm.o
	@echo to build test_auc
	@g++ test_auc.o tools.o ../svm/svm.o -o test_auc

clean:
	rm $(OBJS) test_auc

