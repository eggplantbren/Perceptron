#include "MyConditionalPrior.h"
#include "DNest4/code/DNest4.h"
#include <cmath>

using namespace DNest4;

const Cauchy MyConditionalPrior::c(0.0, 3.0);

MyConditionalPrior::MyConditionalPrior()
{

}

void MyConditionalPrior::from_prior(RNG& rng)
{
    center = c.generate(rng);
    width = 3*rng.rand();
}

double MyConditionalPrior::perturb_hyperparameters(RNG& rng)
{
	double logH = 0.0;

    int which = rng.rand_int(2);
    if(which == 0)
    {
        center = c.cdf(center);
        center += rng.randh();
        wrap(center, 0.0, 1.0);
        center = c.cdf_inverse(center);
    }
    else
    {
        width += 3*rng.randh();
        wrap(width, 0.0, 3.0);
    }

	return logH;
}

double MyConditionalPrior::log_pdf(const std::vector<double>& vec) const
{
    Laplace l(center, width);
	return l.log_pdf(vec[0]);
}

void MyConditionalPrior::from_uniform(std::vector<double>& vec) const
{
    Laplace l(center, width);
    vec[0] = l.cdf_inverse(vec[0]);    
}

void MyConditionalPrior::to_uniform(std::vector<double>& vec) const
{
    Laplace l(center, width);
    vec[0] = l.cdf(vec[0]);
}

void MyConditionalPrior::print(std::ostream& out) const
{
	out<<center<<' '<<width<<' ';
}

