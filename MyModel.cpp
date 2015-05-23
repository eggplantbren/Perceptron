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

Vector MyModel::calculate_output(const Vector& input) const
{
	Vector result;

	// Loop over layers
	for(size_t i=0; i<num_neurons.size()-1; i++)
	{
		// Reshape the weights into a matrix
		Matrix M(num_neurons[i+1], num_neurons[i]);
		const vector< vector<double> >& components = 
			weights[i].get_components();
		int k = 0;
		for(int m=0; m<num_neurons[i+1]; m++)
			for(int n=0; n<num_neurons[i]; n++)
				M(m, n) = components[k++][0];
		// Reshape the biases into a vector
		Vector b(num_neurons[i+1]);
		const vector< vector<double> >& components2 =
			biases[i].get_components();
		for(int m=0; m<num_neurons[i+1]; m++)
			b(i) = components2[i][0];

		// Compute the next layer
		// Linear part
		result = M*input + b;
		// Nonlinear part (not applied to last step)
		if(i != (int)num_neurons.size() - 2)
			for(int m=0; m<result.size(); m++)
				result(i) = tanh(result(i));
	}

	return result;
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

