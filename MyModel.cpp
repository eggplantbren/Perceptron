#include "MyModel.h"
#include "RandomNumberGenerator.h"
#include "Utils.h"
#include <cmath>

using namespace std;
using namespace DNest3;

MyModel::MyModel()
{

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

