#include "MyModel.h"
#include "RandomNumberGenerator.h"
#include "Utils.h"
#include <cmath>

using namespace std;
using namespace DNest3;

MyModel::MyModel()
:weights()
{
	// One input to ten hidden
	weights.push_back(RJObject<MyDistribution>(1, 10, true, MyDistribution()));

	// Ten hidden to one output
	weights.push_back(RJObject<MyDistribution>(1, 10, true, MyDistribution()));
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

void MyModel::print(std::ostream& out) const
{
	// Activity

//	weights.print(out);
}

string MyModel::description() const
{
	return string("");
}

