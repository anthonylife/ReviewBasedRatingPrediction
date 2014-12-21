CC = g++ -std=c++0x
CFLAGS = -Wall -O3 -fopenmp -Wcpp 

all: ${train_target}

jmars: main.cpp ../utils.hpp ../LBFGSCPP.h jmars.hpp corpus.hpp
	$(CC) $(CFLAGS) -o run main.cpp ../LBFGSCPP.cpp

clean:
	rm run
