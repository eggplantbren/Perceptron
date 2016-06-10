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
    sign_control = -1.0 + 2*rng.rand();
}

double Anything::perturb(DNest4::RNG& rng)
{
    double logH = 0.0;

    int which = rng.rand_int(2);

    if(which == 0)
    {
        magnitude = log(magnitude);
        logH += cauchy.perturb(magnitude, rng);
        magnitude = exp(magnitude);
    }
    else
    {
        sign_control += 2*rng.randh();
        DNest4::wrap(sign_control, -1.0, 1.0);
    }

    return logH;
}

double Anything::get_value() const
{
    return sign_control*magnitude;
}

double Anything::get_magnitude() const
{
    return std::abs(get_value());
}

} // namespace Perceptron

