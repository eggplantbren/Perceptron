#ifndef Perceptron_Perceptron
#define Perceptron_Perceptron

#include "MyConditionalPrior.h"
#include "Data.h"
#include "DNest4/code/DNest4.h"
#include "Anything.h"

namespace Perceptron
{

class Perceptron
{
	private:
        static const DNest4::Cauchy cauchy;

        // Account for "arbitrary" units of inputs and outputs
        std::vector<Anything> input_locations, output_locations;
        std::vector<Anything> input_scales, output_scales;

        DNest4::RJObject<MyConditionalPrior> weights;
        std::vector<Matrix> weights_matrices;

        void make_weights_matrices();
        static double nonlinear_function(double x);

	public:
        // Constructors
        Perceptron();
		Perceptron(const std::initializer_list<unsigned int>& num_hidden);

		// Generate the point from the prior
		void from_prior(DNest4::RNG& rng);

		// Metropolis-Hastings proposals
		double perturb(DNest4::RNG& rng);

		// Print to stream
		void print(std::ostream& out) const;

		// Return string with column information
		std::string description() const;

        Vector calculate_output(const Vector& input) const;
};

} // namespace Perceptron

#endif

