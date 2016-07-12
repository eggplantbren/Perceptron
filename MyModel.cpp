#include "MyModel.h"

namespace Perceptron
{

MyModel::MyModel()
:MyModel{10}
{

}

MyModel::MyModel(const std::initializer_list<unsigned int>& num_hidden)
:mean(num_hidden)
,log_sig(num_hidden)
{

}

void MyModel::from_prior(DNest4::RNG& rng)
{
    mean.from_prior(rng);
    log_sig.from_prior(rng);
}


double MyModel::perturb(DNest4::RNG& rng)
{
    double logH = 0.0;

    int which = rng.rand_int(2);

    if(which == 0)
        logH += mean.perturb(rng);
    else
        logH += log_sig.perturb(rng);

    return logH;
}

double MyModel::log_likelihood() const
{
    double logL = 0.0;

    const auto& inputs = Data::get_instance().get_inputs();
    const auto& outputs = Data::get_instance().get_outputs();

    double var;
    for(size_t i=0; i<inputs.size(); ++i)
    {
        // Predict the output
        Vector result = mean.calculate_output(inputs[i]);
        Vector logsigma = log_sig.calculate_output(inputs[i]);

        for(int j=0; j<result.size(); ++j)
        {
            var = exp(2*logsigma[j]);
            logL += -0.5*log(2*M_PI*var)
                    -0.5*pow(outputs[i][j] - result[j], 2)/var;
        }
    }

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

