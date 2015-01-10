CC = g++ -std=c++0x
CFLAGS = -Wall -O3 -fopenmp -Wcpp 

all: ${train_target}

cmr: main.cpp ../utils.hpp ../LBFGSCPP.h cmr.hpp corpus.hpp
	$(CC) $(CFLAGS) -o run main.cpp ../LBFGSCPP.cpp

clean:
	rm run
