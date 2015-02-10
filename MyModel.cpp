#include "MyModel.h"
#include "RandomNumberGenerator.h"
#include "Utils.h"
#include <cmath>

using namespace std;
using namespace DNest3;

MyModel::MyModel()
:num_layers(3)
,num_neurons(num_layers)
,weights()
{
	// Input layer
	num_neurons[0] = 1;

	// Hidden layer
	num_neurons[1] = 10;

	// Output layer
	num_neurons[2] = 1;

	// Weights
	for(int i=0; i<(num_layers - 1); i++)
		weights.push_back(RJObject<MyDistribution>(1, num_neurons[i]*num_neurons[i+1], true, MyDistribution()));
}

void MyModel::fromPrior()
{
	for(size_t i=0; i<weights.size(); i++)
		weights[i].fromPrior();
}

double MyModel::perturb()
{
	double logH = 0.;

	int which = randInt(weights.size());
	logH += weights[which].perturb();

	return logH;
}

double MyModel::logLikelihood() const
{
	return 0.;
}

vector<double> MyModel::compute_output(const vector<double>& input) const
{
	return vector<double>();
}

void MyModel::print(std::ostream& out) const
{
	// Activity

//	weights.print(out);
}

string MyModel::description() const
{
	return string("");
}

