#include "MyDistribution.h"
#include "RandomNumberGenerator.h"
#include "Utils.h"
#include <cmath>
#include <gsl/gsl_cdf.h>

using namespace DNest3;

MyDistribution::MyDistribution()
{

}

void MyDistribution::fromPrior()
{
	sigma = exp(tan(M_PI*(0.97*randomU() - 0.485)));
}

double MyDistribution::perturb_parameters()
{
	double logH = 0.;

	sigma = log(sigma);
	sigma = (atan(sigma)/M_PI + 0.485)/0.97;
	sigma += randh();
	wrap(sigma, 0., 1.);
	sigma = tan(M_PI*(0.97*sigma - 0.485));
	sigma = exp(sigma);


	return logH;
}

// vec[0] = central position
// vec[1] = log_width
// vec[2] = log_weight

double MyDistribution::log_pdf(const std::vector<double>& vec) const
{
	double logp = 0.;

	logp += -log(sigma) - 0.5*pow(vec[0]/sigma, 2);

	return logp;
}

void MyDistribution::from_uniform(std::vector<double>& vec) const
{
	vec[0] = sigma*gsl_cdf_ugaussian_Pinv(vec[0]);
}

void MyDistribution::to_uniform(std::vector<double>& vec) const
{
	vec[0] = gsl_cdf_ugaussian_P(vec[0]/sigma);
}

void MyDistribution::print(std::ostream& out) const
{
	out<<sigma<<' ';
}

