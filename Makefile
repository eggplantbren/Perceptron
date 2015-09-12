#CFLAGS = -O2 -DARMA_NO_DEBUG -DNDEBUG -Wall -Wextra -ansi -pedantic
CFLAGS = -m64 -O3 -flto -march=native -funroll-loops -DARMA_NO_DEBUG -DNDEBUG -Wall -Wextra -ansi -pedantic
LIBS = -lrjobject -ldnest3 -lgsl -lgslcblas -lboost_thread -lboost_system

default:
	g++ $(CFLAGS) -c *.cpp
	g++ -o main *.o $(LIBS)
	rm -f *.o

