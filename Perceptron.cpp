#include "Perceptron.h"

namespace Perceptron
{

Perceptron::Perceptron(const std::initializer_list<unsigned int>& num_hidden)
{
    // Number of nodes in each layer INCLUDING input and output layers
    std::vector<size_t> num_nodes;
    num_nodes.push_back(Data::get_instance().get_dim_inputs());
    for(auto n: num_hidden)
        num_nodes.push_back(n);
    num_nodes.push_back(Data::get_instance().get_dim_outputs());

    // Calculate total number of weights and biases needed
    size_t num_weights = 0;
    for(size_t i=1; i<num_nodes.size(); ++i)
    {
        weights.push_back(DNest4::RJObject<MyConditionalPrior>(1,
                      num_nodes[i]*num_nodes[i-1], true, MyConditionalPrior()));
        weights_matrices.push_back(Matrix(num_nodes[i], num_nodes[i-1]));
        num_weights += num_nodes[i]*num_nodes[i-1];
    }

    size_t num_biases = 0;
    for(auto n: num_hidden)
    {
        biases.push_back(DNest4::RJObject<MyConditionalPrior>(1, n, true,
                                            MyConditionalPrior()));
        bias_vectors.push_back(Vector(n));
        num_biases += n;
    }
    biases.push_back(DNest4::RJObject<MyConditionalPrior>(1,
                                Data::get_instance().get_dim_outputs(), true,
                                            MyConditionalPrior()));
    bias_vectors.push_back(Vector(Data::get_instance().get_dim_outputs()));
    num_biases += Data::get_instance().get_dim_outputs();
}


void Perceptron::from_prior(DNest4::RNG& rng)
{
    for(size_t i=0; i<weights.size(); ++i)
    {
        weights[i].from_prior(rng);
        make_weights_matrix(i);
    }

    for(size_t i=0; i<biases.size(); ++i)
    {
        biases[i].from_prior(rng);
        make_bias_vector(i);
    }
}


double Perceptron::perturb(DNest4::RNG& rng)
{
    double logH = 0.0;

    if(rng.rand() <= 0.5)
    {
        size_t which = rng.rand_int(weights.size());
        logH += weights[which].perturb(rng);
        make_weights_matrix(which);
    }
    else
    {
        size_t which = rng.rand_int(biases.size());
        logH += biases[which].perturb(rng);
        make_bias_vector(which);
    }

    return logH;
}

void Perceptron::make_weights_matrix(size_t which)
{
    // Put weights into a matrix
    const auto& components = weights[which].get_components();
    auto& weights_matrix = weights_matrices[which];

    unsigned int index = 0;
    for(int i=0; i<weights_matrix.rows(); ++i)
        for(int j=0; j<weights_matrix.cols(); ++j)
            weights_matrix(i, j) = components[index++][0];
}

void Perceptron::make_bias_vector(size_t which)
{
    // Put weights in matrices
    const auto& components = biases[which].get_components();
    auto& bias_vector = bias_vectors[which];

    unsigned int index = 0;
    for(int i=0; i<bias_vector.size(); ++i)
        bias_vector(i) = components[index++][0];
}

void Perceptron::print(std::ostream& out) const
{
//    for(const auto& x: input_locations)
//        out<<x.get_value()<<' ';
//    for(const auto& x: output_locations)
//        out<<x.get_value()<<' ';

//    for(const auto& x: input_scales)
//        out<<x.get_magnitude()<<' ';
//    for(const auto& x: output_scales)
//        out<<x.get_magnitude()<<' ';

//    weights.print(out);
//    out<<sigma<<' ';

    Vector input(1);
    for(double x=-20.0; x<= 20.0000001; x += 0.02)
    {
        input[0] = x;
        auto output = calculate_output(input);
        out<<output(0)<<' ';
    }
}

Vector Perceptron::calculate_output(const Vector& input) const
{
    // Apply the neural net function on the input
    Vector result = input;

    for(size_t i=0; i<weights_matrices.size(); ++i)
    {
        result = weights_matrices[i]*result + bias_vectors[i];

        // Apply the nonlinear function unless this is the output layer
        if(i != (weights_matrices.size() -1))
        {
            for(int j=0; j<result.size(); ++j)
                result[j] = nonlinear_function(result[j]);
        }
    }

    return result;
}

std::string Perceptron::description() const
{
    return std::string("");
}

double Perceptron::nonlinear_function(double x)
{
    return (x < 0.0)?(0.0):(x);
}

} // namespace Perceptron

