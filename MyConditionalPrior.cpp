#include "MyConditionalPrior.h"
#include "DNest4/code/DNest4.h"
#include <cmath>

using namespace DNest4;

namespace Perceptron
{

MyConditionalPrior::MyConditionalPrior()
{

}

void MyConditionalPrior::from_prior(RNG& rng)
{
    const Cauchy cauchy(0.0, 5.0);
    do
    {
        sigma = cauchy.generate(rng);
    }while(std::abs(sigma) >= 50.0);
    sigma = exp(sigma);
}

double MyConditionalPrior::perturb_hyperparameters(RNG& rng)
{
    const Cauchy cauchy(0.0, 5.0);

	double logH = 0.0;

    sigma = log(sigma);
    logH += cauchy.perturb(sigma, rng);
    if(std::abs(sigma) >= 50.0)
    {
        sigma = 1.0;
        return -1E300;
    }
    sigma = exp(sigma);

	return logH;
}

double MyConditionalPrior::log_pdf(const std::vector<double>& vec) const
{
    Laplace l(0, sigma);
	return l.log_pdf(vec[0]);
}

void MyConditionalPrior::from_uniform(std::vector<double>& vec) const
{
    Laplace l(0, sigma);
    vec[0] = l.cdf_inverse(vec[0]);
}

void MyConditionalPrior::to_uniform(std::vector<double>& vec) const
{
    Laplace l(0, sigma);
    vec[0] = l.cdf(vec[0]);
}

void MyConditionalPrior::print(std::ostream& out) const
{

}

} // namespace Perceptron

