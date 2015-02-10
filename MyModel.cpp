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
	num_neurons[0] = 100;

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
	vector< vector<double> > output(num_layers);
	output[0] = input;

	// Compute each layer (except input layer)
	for(int i=1; i<num_layers; i++)
	{
		const vector< vector<double> >& components = weights[i-1].get_components();

		// Calculate activations for this layer
		output[i].assign(num_neurons[i], 0.);
		for(int j=0; j<num_neurons[i]; j++)
		{
			for(int k=0; k<num_neurons[i-1]; k++)
				output[i][j] += components[j*num_neurons[i-1] + k][0]*output[i-1][k];
			output[i][j] = tanh(output[i][j]);
		}
	}

	return output.back();
}

void MyModel::print(std::ostream& out) const
{
	vector<double> input(101);
	for(int i=0; i<101; i++)
		input[i] = 0.01*i;

	vector<double> output = compute_output(input);
	for(size_t i=0; i<output.size(); i++)
		out<<output[i]<<' ';
}

string MyModel::description() const
{
	return string("");
}

