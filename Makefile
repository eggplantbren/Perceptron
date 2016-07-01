#CFLAGS = -O2 -DARMA_NO_DEBUG -DNDEBUG -Wall -Wextra -ansi -pedantic
CFLAGS = -std=c++11 -O3 -Wall -pedantic
LIBS = -ldnest4 -lpthread

default:
	g++ -I$(DNEST4_PATH) $(CFLAGS) -c *.cpp
	g++ -L$(DNEST4_PATH)/DNest4/code -o main *.o $(LIBS)
	rm -f *.o

