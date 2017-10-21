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
        std::vector<DNest4::RJObject<MyConditionalPrior>> weights;
        std::vector<DNest4::RJObject<MyConditionalPrior>> biases;
        std::vector<Matrix> weights_matrices;
        std::vector<Vector> bias_vectors;

        void make_weights_matrix(size_t which);
        void make_bias_vector(size_t which);
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

