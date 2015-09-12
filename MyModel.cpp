#include "MyModel.h"
#include "RandomNumberGenerator.h"
#include "Utils.h"
#include <cmath>

using namespace std;
using namespace DNest3;

MyModel::MyModel()
:num_neurons(3)
{
	num_neurons[0] = Data::get_instance().get_inputs()[0].size();
	num_neurons[1] = 5;
	num_neurons[2] = Data::get_instance().get_outputs()[0].size();

	for(size_t i=0; i<num_neurons.size()-1; i++)
	{
		weights.push_back(RJObject<MyDistribution>(1,
					num_neurons[i]*num_neurons[i+1],
					true, MyDistribution()));
		biases.push_back(RJObject<MyDistribution>(1,
					num_neurons[i+1],
					true, MyDistribution()));
	}
}

void MyModel::fromPrior()
{
	for(size_t i=0; i<weights.size(); i++)
		weights[i].fromPrior();
	for(size_t i=0; i<biases.size(); i++)
		biases[i].fromPrior();
	sigma = exp(tan(M_PI*(0.97*randomU() - 0.485)));
}

double MyModel::perturb()
{
	double logH = 0.;

	int which = randInt(3);
	if(which == 0)
		logH += weights[randInt(weights.size())].perturb();
	else if(which == 1)
		logH += biases[randInt(biases.size())].perturb();
	else
	{
		sigma = log(sigma);
		sigma = (atan(sigma)/M_PI + 0.485)/0.97;
		sigma += randh();
		wrap(sigma, 0., 1.);
		sigma = tan(M_PI*(0.97*sigma - 0.485));
		sigma = exp(sigma);
	}

	return logH;
}

Vector MyModel::calculate_output(const Vector& input) const
{
	Vector result = input;

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
			b(m) = components2[m][0];

		// Compute the next layer
		// Linear part
		result = M*result + b;

		// Nonlinear part (not applied to last step)
		if((int)i != (int)num_neurons.size() - 2)
			for(int m=0; m<result.size(); m++)
				result(m) = tanh(result(m));
	}

	return result;
}

double MyModel::logLikelihood() const
{
	double logL = 0.;

	// Get the data
	const vector<Vector>& inputs = Data::get_instance().get_inputs();
	const vector<Vector>& outputs = Data::get_instance().get_outputs();

	Vector output;
	for(size_t i=0; i<inputs.size(); i++)
	{
		output = calculate_output(inputs[i]);
		for(int j=0; j<output.size(); j++)
		{
			logL += -log(sigma) -
				0.5*pow((outputs[i](j) - output(j))/sigma, 2);
		}
	}

	return logL;
}

void MyModel::print(std::ostream& out) const
{
	Vector input(2);
	input(0) = 2.;
	input(1) = -1.;
	Vector output = calculate_output(input);
	out<<(output(0) + sigma*randn())<<' '<<0<<endl;
}

string MyModel::description() const
{
	return string("");
}

