#include "MyModel.h"
#include "RandomNumberGenerator.h"
#include "Utils.h"
#include <cmath>

using namespace std;
using namespace DNest3;

MyModel::MyModel()
:weights(1, 100, true, MyDistribution())
{

}

void MyModel::fromPrior()
{
	weights.fromPrior();
}

double MyModel::perturb()
{
	double logH = 0.;

	logH += weights.perturb();

	return logH;
}

double MyModel::logLikelihood() const
{
	return 0.;
}

void MyModel::print(std::ostream& out) const
{
	weights.print(out);
}

string MyModel::description() const
{
	return string("");
}

