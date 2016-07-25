#include "Perceptron.h"

namespace Perceptron
{

const DNest4::Cauchy Perceptron::cauchy(0.0, 5.0);

Perceptron::Perceptron()
:Perceptron{10}
{

}

Perceptron::Perceptron(const std::initializer_list<unsigned int>& num_hidden)
:input_locations(Data::get_instance().get_dim_inputs())
,output_locations(Data::get_instance().get_dim_outputs())
,input_scales(Data::get_instance().get_dim_inputs())
,output_scales(Data::get_instance().get_dim_outputs())
,weights(1, 1, true, MyConditionalPrior()) // Placeholder
{
    // Number of nodes in each layer INCLUDING input and output layers
    std::vector<size_t> num_nodes;
    num_nodes.push_back(Data::get_instance().get_dim_inputs());
    for(auto n: num_hidden)
        num_nodes.push_back(n);
    num_nodes.push_back(Data::get_instance().get_dim_outputs());

    size_t num_weights=0; // Total number of weights needed
    for(size_t i=1; i<num_nodes.size(); ++i)
    {
        weights_matrices.push_back(Matrix(num_nodes[i], num_nodes[i-1]));
        num_weights += num_nodes[i]*num_nodes[i-1];
    }

    // Initialise weights object
    weights = DNest4::RJObject<MyConditionalPrior>(1, num_weights,
                                                true, MyConditionalPrior());
}


void Perceptron::from_prior(DNest4::RNG& rng)
{
    for(auto& x: input_locations)
        x.from_prior(rng);
    for(auto& x: output_locations)
        x.from_prior(rng);
    for(auto& x: input_scales)
        x.from_prior(rng);
    for(auto& x: output_scales)
        x.from_prior(rng);

    weights.from_prior(rng);
    make_weights_matrices();
}


double Perceptron::perturb(DNest4::RNG& rng)
{
    double logH = 0.0;

    if(rng.rand() <= 0.5)
    {
        int which = rng.rand_int(4);

        if(which == 0)
        {
            logH += input_locations[rng.rand_int(input_locations.size())]
                                    .perturb(rng);
        }
        else if(which == 1)
        {
            logH += output_locations[rng.rand_int(output_locations.size())]
                                    .perturb(rng);
        }
        if(which == 2)
        {
            logH += input_scales[rng.rand_int(input_scales.size())]
                                    .perturb(rng);
        }
        else
        {
            logH += output_scales[rng.rand_int(output_scales.size())]
                                    .perturb(rng);
        }
    }
    else
    {
        logH += weights.perturb(rng, false);
        make_weights_matrices();
    }


    return logH;
}

void Perceptron::make_weights_matrices()
{
    // Put weights in matrices
    const auto& components = weights.get_components();

    unsigned int index = 0;
    for(auto& weights_matrix: weights_matrices)
    {
        for(int i=0; i<weights_matrix.rows(); ++i)
            for(int j=0; j<weights_matrix.cols(); ++j)
                weights_matrix(i, j) = components[index++][0];
    }
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
    for(double x=-10.0; x<= 10.0000001; x += 0.01)
    {
        input[0] = x;
        auto output = calculate_output(input);
        out<<output(0)<<' ';
    }
}

Vector Perceptron::calculate_output(const Vector& input) const
{
    // Standardize the input of the example
    Vector standardized_input = input;
    for(int j=0; j<standardized_input.size(); ++j)
    {
        standardized_input[j] -= input_locations[j].get_value();
        standardized_input[j] /= input_scales[j].get_magnitude();
    }

    // Run the neural net on the standardized input
    Vector result = standardized_input;

    for(size_t i=0; i<weights_matrices.size(); ++i)
    {
        result = weights_matrices[i]*result;
        for(int j=0; j<result.size(); ++j)
            result[j] = nonlinear_function(result[j]);
    }

    // De-standardize the output
    for(int j=0; j<result.size(); ++j)
    {
        result[j] *= output_scales[j].get_magnitude();
        result[j] += output_locations[j].get_value();
    }

    return result;
}

std::string Perceptron::description() const
{
    return std::string("");
}

double Perceptron::nonlinear_function(double x)
{
    return tanh(x);
}

} // namespace Perceptron

