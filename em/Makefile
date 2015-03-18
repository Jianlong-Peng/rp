GXX = g++
CFLAGS = -Wall -O2
CFLAGS_P = -Wall -O2 -DNTHREAD

OBJS = ../svm/svm.o ../svm/svmtools.o ../utilities/tools.o

all: em em_p

#tools_p.o: tools.cpp
#	@echo to build $@
#	@$(GXX) $(CFLAGS_P) -c $< -o $@

%_p.o: %.cpp
	@echo to build $@
	@$(GXX) $(CFLAGS_P) -c $< -o $@

%.o: %.cpp
	@echo to build $@
	@$(GXX) $(CFLAGS) -c $<

em_p: main_p.o tools_p.o $(OBJS)
	@echo to build em_p
	@g++ $(OBJS) tools_p.o main_p.o -o em_p -lpthread

em: main.o tools.o $(OBJS)
	@echo to build em
	@g++ $(OBJS) tools.o main.o -o em

clean:
	rm *.o em em_p

