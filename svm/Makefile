CFLAGS = -Wall -O2
#CFLAGS = -Wall -g

OBJS = svm.o svmtools.o

all: test_grid grid_search

%.o: %.cpp
	@echo to build $@
	@g++ $(CFLAGS) $<

test_grid: test_grid.o $(OBJS)
	@echo to build test_grid
	@g++ $(OBJS) test_grid.o -o test_grid

grid_search: grid_search.o $(OBJS)
	@echo to build grid_search
	@g++ $(OBJS) grid_search.o -o grid_search
