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
    mu.from_prior(rng);

    const Cauchy cauchy(0.0, 5.0);
    do
    {
        sigma = cauchy.generate(rng);
    }while(std::abs(sigma) >= 50.0);
    sigma = exp(sigma);
}

double MyConditionalPrior::perturb_hyperparameters(RNG& rng)
{
	double logH = 0.0;

    if(rng.rand() <= 0.5)
    {
        logH += mu.perturb(rng);
    }
    else
    {
        const Cauchy cauchy(0.0, 5.0);

        sigma = log(sigma);
        logH += cauchy.perturb(sigma, rng);
        if(std::abs(sigma) >= 50.0)
        {
            sigma = 1.0;
            return -1E300;
        }
        sigma = exp(sigma);
    }

	return logH;
}

double MyConditionalPrior::log_pdf(const std::vector<double>& vec) const
{
    Laplace l(mu.get_value(), sigma);
	return l.log_pdf(vec[0]);
}

void MyConditionalPrior::from_uniform(std::vector<double>& vec) const
{
    Laplace l(mu.get_value(), sigma);
    vec[0] = l.cdf_inverse(vec[0]);
}

void MyConditionalPrior::to_uniform(std::vector<double>& vec) const
{
    Laplace l(mu.get_value(), sigma);
    vec[0] = l.cdf(vec[0]);
}

void MyConditionalPrior::print(std::ostream& out) const
{

}

} // namespace Perceptron

