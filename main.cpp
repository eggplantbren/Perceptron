#include <iostream>
#include "Data.h"
#include "MyModel.h"
#include "DNest4/code/DNest4.h"

using namespace std;

int main(int argc, char** argv)
{
	Data::get_instance().load("clocks.txt");
    DNest4::start<MyModel>(argc, argv);
	return 0;
}

