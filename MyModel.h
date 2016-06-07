#ifndef Perceptron_MyModel
#define Perceptron_MyModel

#include "MyConditionalPrior.h"
#include "Data.h"
#include "DNest4/code/DNest4.h"

class MyModel
{
	private:

        std::vector<double> calculate_output
                (const std::vector<double>& input) const;

	public:
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

#endif

