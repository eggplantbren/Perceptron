#CFLAGS = -O2 -DARMA_NO_DEBUG -DNDEBUG -Wall -Wextra -ansi -pedantic
DNEST4_PATH=/home/brewer/Projects
EIGEN_PATH=/usr/include/eigen3
CFLAGS = -std=c++11 -O3 -Wall -pedantic -march=native
LIBS = -ldnest4 -lpthread

default:
	g++ -I$(DNEST4_PATH) -I$(EIGEN_PATH) $(CFLAGS) -c *.cpp
	g++ -L$(DNEST4_PATH)/DNest4/code -o main *.o $(LIBS)
	rm -f *.o

