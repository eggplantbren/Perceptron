#include "MyModel.h"
#include "RandomNumberGenerator.h"
#include "Utils.h"
#include <cmath>

using namespace std;
using namespace DNest3;

MyModel::MyModel()
:num_neurons(3)
{
	num_neurons[0] = 2;
	num_neurons[1] = 5;
	num_neurons[2] = 1;

	for(size_t i=0; i<num_neurons.size(); i++)
	{
		weights.push_back(RJObject<MyDistribution>(1,
					num_neurons[i-1]*num_neurons[i],
					true, MyDistribution()));
		biases.push_back(RJObject<MyDistribution>(1,
					num_neurons[i],
					true, MyDistribution()));
	}
}

void MyModel::fromPrior()
{

}

double MyModel::perturb()
{
	double logH = 0.;

	return logH;
}

double MyModel::logLikelihood() const
{
	return 0.;
}

void MyModel::print(std::ostream& out) const
{
}

string MyModel::description() const
{
	return string("");
}

