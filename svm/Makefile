GXX = icc
CFLAGS = -Wall -O2
#CFLAGS = -Wall -g

OBJS = svm.o svmtools.o
EXE = test_grid grid_search

all: $(OBJS) $(EXE)

%.o: %.cpp
	@echo to build $@
	@$(GXX) $(CFLAGS) -c $<

test_grid: test_grid.o $(OBJS)
	@echo to build test_grid
	@$(GXX) $(OBJS) test_grid.o -o test_grid

grid_search: grid_search.o $(OBJS)
	@echo to build grid_search
	@$(GXX) $(OBJS) grid_search.o -o grid_search

clean:
	rm *.o $(EXE)

