#ifndef Perceptron_Anything
#define Perceptron_Anything

#include "DNest4/code/DNest4.h"

namespace Perceptron
{

/*
* Convenience wrapper for a prior over doubles
* that covers many orders of magnitude and can be negative.
*/
class Anything
{
    private:
        double magnitude;
        double sign_control;

        // A Cauchy distribution will help
        static const DNest4::Cauchy cauchy;

    public:
        Anything();
        void from_prior(DNest4::RNG& rng);
        double perturb(DNest4::RNG& rng);
        double get_value() const;
        double get_magnitude() const;
};

} // namespace Perceptron

#endif

