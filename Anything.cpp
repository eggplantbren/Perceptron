#include "Anything.h"

namespace Perceptron
{

const DNest4::Cauchy Anything::cauchy(0.0, 5.0);

Anything::Anything()
{

}

void Anything::from_prior(DNest4::RNG& rng)
{
    magnitude = exp(cauchy.generate(rng));
    if(rng.rand() <= 0.5)
        sign = -1;
    else
        sign = 1;
}

double Anything::perturb(DNest4::RNG& rng)
{
    double logH = 0.0;

    magnitude = log(magnitude);
    logH += cauchy.perturb(magnitude, rng);
    magnitude = exp(magnitude);

    if(rng.rand() <= 0.2)
        sign *= -1;

    return logH;
}

double Anything::get_value() const
{
    return sign*magnitude;
}

double Anything::get_magnitude() const
{
    return magnitude;
}

} // namespace Perceptron

