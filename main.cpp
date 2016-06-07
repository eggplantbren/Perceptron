#include <iostream>
#include "Data.h"

using namespace std;

int main(int argc, char** argv)
{
	Data::get_instance().load("clocks.txt");

	return 0;
}

