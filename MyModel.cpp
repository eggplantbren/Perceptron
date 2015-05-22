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
	for(size_t i=0; i<weights.size(); i++)
		weights[i].fromPrior();
	for(size_t i=0; i<biases.size(); i++)
		biases[i].fromPrior();
}

double MyModel::perturb()
{
	double logH = 0.;

	int which = randInt(2);
	if(which == 0)
		logH += weights[randInt(weights.size())].perturb();
	else
		logH += biases[randInt(biases.size())].perturb();

	return logH;
}

double MyModel::logLikelihood() const
{
	double logL = 0.;

	return logL;
}

void MyModel::print(std::ostream& out) const
{
}

string MyModel::description() const
{
	return string("");
}

