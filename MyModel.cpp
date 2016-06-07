#include "MyModel.h"

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
}

double MyModel::perturb(DNest4::RNG& rng)
{
    double logH = 0.0;

    logH += weights.perturb(rng);
    make_weights_matrix();

    return logH;
}

void MyModel::make_weights_matrix()
{
    // Put weights in a matrix
    const auto& components = weights.get_components();
    int k = 0;
    for(int i=0; i<Data::get_instance().get_dim_outputs(); ++i)
        for(int j=0; j<Data::get_instance().get_dim_outputs(); ++j)
            weights_matrix(i, j) = components[k++][0];
}

double MyModel::log_likelihood() const
{
    double logL = 0.0;

    return logL;
}

void MyModel::print(std::ostream& out) const
{
    weights.print(out);
}

Vector MyModel::calculate_output(const Vector& input) const
{
    return weights_matrix*input;
}

