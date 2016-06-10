#include "MyModel.h"

namespace Perceptron
{

const DNest4::Cauchy MyModel::cauchy(0.0, 5.0);

MyModel::MyModel()
:input_locations(Data::get_instance().get_dim_inputs())
,output_locations(Data::get_instance().get_dim_outputs())
,input_scales(Data::get_instance().get_dim_inputs())
,output_scales(Data::get_instance().get_dim_outputs())
,weights(1,
         Data::get_instance().get_dim_outputs()*
                Data::get_instance().get_dim_inputs(),
         true,
         MyConditionalPrior())
,weights_matrix(Data::get_instance().get_dim_outputs(),
                Data::get_instance().get_dim_inputs())
{

}

void MyModel::from_prior(DNest4::RNG& rng)
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
    make_weights_matrix();

    sigma = exp(cauchy.generate(rng));
}

double MyModel::perturb(DNest4::RNG& rng)
{
    double logH = 0.0;

    int which = rng.rand_int(6);

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
        logH += input_scales[rng.rand_int(input_scales.size())].perturb(rng);
    }
    else if(which == 3)
    {
        logH += output_scales[rng.rand_int(output_scales.size())].perturb(rng);
    }
    else if(which == 4)
    {
        logH += weights.perturb(rng);
        make_weights_matrix();
    }
    else
    {
        sigma = log(sigma);
        logH += cauchy.perturb(sigma, rng);
        sigma = exp(sigma);
    }

    return logH;
}

void MyModel::make_weights_matrix()
{
    // Put weights in a matrix
    const auto& components = weights.get_components();
    int k = 0;
    for(int i=0; i<Data::get_instance().get_dim_outputs(); ++i)
        for(int j=0; j<Data::get_instance().get_dim_inputs(); ++j)
            weights_matrix(i, j) = components[k++][0];
}

double MyModel::log_likelihood() const
{
    double logL = 0.0;

    const auto& inputs = Data::get_instance().get_inputs();
    const auto& outputs = Data::get_instance().get_outputs();

    for(size_t i=0; i<inputs.size(); ++i)
    {
        // Predict the output
        Vector result = calculate_output(inputs[i]);

        for(int j=0; j<result.size(); ++j)
        {
            double var = pow(sigma*output_scales[j].get_magnitude(), 2);
            logL += -0.5*log(2*M_PI*var)
                    -0.5*pow(outputs[i][j] - result[j], 2)/var;
        }
    }

    return logL;
}

void MyModel::print(std::ostream& out) const
{
    for(const auto& x: input_locations)
        out<<x.get_value()<<' ';
    for(const auto& x: output_locations)
        out<<x.get_value()<<' ';

    for(const auto& x: input_scales)
        out<<x.get_magnitude()<<' ';
    for(const auto& x: output_scales)
        out<<x.get_magnitude()<<' ';

    weights.print(out);
    out<<sigma<<' ';
}

Vector MyModel::calculate_output(const Vector& input) const
{
    // Standardize the input of the example
    Vector standardized_input = input;
    for(int j=0; j<standardized_input.size(); ++j)
    {
        standardized_input[j] -= input_locations[j].get_value();
        standardized_input[j] /= input_scales[j].get_magnitude();
    }

    // Run the neural net on the standardized input
    Vector result = weights_matrix*input;
    for(int i=0; i<result.size(); ++i)
        result[i] = nonlinear_function(result[i]);

    // De-standardize the output
    for(int j=0; j<result.size(); ++j)
    {
        result[j] *= output_scales[j].get_magnitude();
        result[j] += output_locations[j].get_value();
    }

    return result;
}

std::string MyModel::description() const
{
    return std::string("");
}

double MyModel::nonlinear_function(double x)
{
    if(x < -1.0)
        return -1.0;
    if(x > 1.0)
        return 1.0;
    return x;
}

} // namespace Perceptron

