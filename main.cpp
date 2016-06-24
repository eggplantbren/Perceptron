#include <iostream>
#include "Data.h"
#include "MyModel.h"
#include "Anything.h"
#include "DNest4/code/DNest4.h"

using namespace Perceptron;
using namespace std;

int main(int argc, char** argv)
{
    Data::get_instance().load("fake_data.txt");
    DNest4::start<MyModel>(argc, argv);
	return 0;
}

