GXX = g++
CFLAGS = -Wall -O2

OBJS = ../svm/svm.o ../svm/svmtools.o ../partition/tools.o tools.o

all: tools.o main.o em

%.o: %.cpp
	@echo to build $@
	@$(GXX) $(CFLAGS) -c $<

em: main.o $(OBJS)
	@echo to build em
	@g++ $(OBJS) main.o -o em

clean:
	rm *.o em
