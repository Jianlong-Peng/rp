CFLAGS = -Wall -O2
#CFLAGS = -Wall -g

all: svm.o svmtools.o

%.o: %.cpp
	@echo to build $@
	@g++ $(CFLAGS) $<
