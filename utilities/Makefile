GA_ROOT = /home/xmluo/jlpeng/program/ga-svm/galib247
CFLAGS = -I$(GA_ROOT) -Wall -O2
#CFLAGS = -I$(GA_ROOT) -Wall -g
CFLAGS_P = -I$(GA_ROOT) -Wall -O2 -DNTHREAD
#CFLAGS_P = -I$(GA_ROOT) -Wall -g -DNTHREAD
#LDFLAGS = -L$(GA_ROOT)/ga -lga

OBJS = tools.o extern_tools.o operators.o

all: $(OBJS)

%.o: %.cpp
	@echo to build $@
	@g++ $(CFLAGS) $< -c

clean:
	rm $(OBJS)

