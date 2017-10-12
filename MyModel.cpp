#include "MyModel.h"
#include <cmath>

namespace Perceptron
{

MyModel::MyModel()
:MyModel{10}
{

}

MyModel::MyModel(const std::initializer_list<unsigned int>& num_hidden)
:mean(num_hidden)
{

}

void MyModel::from_prior(DNest4::RNG& rng)
{
    mean.from_prior(rng);
}


double MyModel::perturb(DNest4::RNG& rng)
{
    double logH = 0.0;

    logH += mean.perturb(rng);

    return logH;
}

double MyModel::log_likelihood() const
{
    double logL = 0.0;

    const auto& inputs = Data::get_instance().get_inputs();
    const auto& outputs = Data::get_instance().get_outputs();

    double C = 0.5*log(2.0*M_PI);
    for(size_t i=0; i<inputs.size(); ++i)
    {
        // Predict the output
        Vector result = mean.calculate_output(inputs[i]);

        for(int j=0; j<result.size(); ++j)
        {
            logL += -C - 0.5*pow(outputs[i][j] - result[j], 2);
        }
    }

    if(std::isnan(logL) || std::isinf(logL))
        logL = -1E300;

    return logL;
}

void MyModel::print(std::ostream& out) const
{
    mean.print(out);
}
std::string MyModel::description() const
{
    return std::string("");
}

} // namespace Perceptron

