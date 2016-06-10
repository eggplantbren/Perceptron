#include "MyModel.h"

namespace Perceptron
{

const DNest4::Cauchy MyModel::cauchy(0.0, 5.0);

MyModel::MyModel()
:weights(1,
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
    weights.from_prior(rng);
    make_weights_matrix();

    sigma = exp(cauchy.generate(rng));
}

double MyModel::perturb(DNest4::RNG& rng)
{
    double logH = 0.0;

    int which = rng.rand_int(2);

    if(which == 0)
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
        Vector result = calculate_output(inputs[i]);
        logL += -0.5*log(2*M_PI) - log(sigma)
                    - 0.5*pow((outputs[i][0] - result[0])/sigma, 2);
    }

    return logL;
}

void MyModel::print(std::ostream& out) const
{
    weights.print(out);
    out<<sigma<<' ';
}

Vector MyModel::calculate_output(const Vector& input) const
{
    return weights_matrix*input;
}

std::string MyModel::description() const
{
    return std::string("");
}

} // namespace Perceptron

