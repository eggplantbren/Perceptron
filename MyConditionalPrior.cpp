#include "MyConditionalPrior.h"
#include "DNest4/code/DNest4.h"
#include <cmath>

using namespace DNest4;

MyConditionalPrior::MyConditionalPrior()
{

}

void MyConditionalPrior::from_prior(RNG& rng)
{

}

double MyConditionalPrior::perturb_hyperparameters(RNG& rng)
{
	double logH = 0.0;

	return logH;
}

double MyConditionalPrior::log_pdf(const std::vector<double>& vec) const
{
    Laplace l;
	return l.log_pdf(vec[0]);
}

void MyConditionalPrior::from_uniform(std::vector<double>& vec) const
{
    Laplace l;
    vec[0] = l.cdf_inverse(vec[0]);    
}

void MyConditionalPrior::to_uniform(std::vector<double>& vec) const
{
    Laplace l;
    vec[0] = l.cdf(vec[0]);
}

void MyConditionalPrior::print(std::ostream& out) const
{

}

