#ifndef Perceptron_MyModel
#define Perceptron_MyModel

#include "MyConditionalPrior.h"
#include "Data.h"
#include "DNest4/code/DNest4.h"
#include "Anything.h"

namespace Perceptron
{

class MyModel
{
	private:
        static const DNest4::Cauchy cauchy;

        std::vector<Anything> input_scales, output_scales;

        DNest4::RJObject<MyConditionalPrior> weights;
        Matrix weights_matrix;

        double sigma;

        void make_weights_matrix();
        Vector calculate_output(const Vector& input) const;

	public:
        // Constructor
		MyModel();

		// Generate the point from the prior
		void from_prior(DNest4::RNG& rng);

		// Metropolis-Hastings proposals
		double perturb(DNest4::RNG& rng);

		// Likelihood function
		double log_likelihood() const;

		// Print to stream
		void print(std::ostream& out) const;

		// Return string with column information
		std::string description() const;
};

} // namespace Perceptron

#endif

