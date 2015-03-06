CFLAGS = -Wall -O2
#CFLAGS = -Wall -g

all: svm.o svmtools.o test_grid

%.o: %.cpp
	@echo to build $@
	@g++ $(CFLAGS) $<

test_grid: test_grid.o svmtools.o
	@echo to build test_grid
	@g++ svmtools.o test_grid.o -o test_grid

