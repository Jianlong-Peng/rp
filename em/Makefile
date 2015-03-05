GXX = g++
CFLAGS = -Wall -O2

all:

%.o: %.cpp ../partition/tools.o ../partition/svm.o
	$(GXX) $(CFLAGS) $<

