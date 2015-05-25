#include "MyDistribution.h"
#include "RandomNumberGenerator.h"
#include "Utils.h"
#include <cmath>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_sf_gamma.h>

using namespace DNest3;

MyDistribution::MyDistribution()
{

}

void MyDistribution::fromPrior()
{
	mu = 1000.*tan(M_PI*(randomU() - 0.5));
	sigma = exp(tan(M_PI*(0.97*randomU() - 0.485)));
	nu = exp(log(1.) + log(1E3)*randomU());
}

double MyDistribution::perturb_parameters()
{
	double logH = 0.;

	int which = randInt(3);

	if(which == 0)
	{
		mu = atan(mu/1000.)/M_PI + 0.5;
		mu += randh();
		wrap(mu, 0., 1.);
		mu = 1000.*tan(M_PI*(mu - 0.5));
	}
	else if(which == 1)
	{
		sigma = log(sigma);
		sigma = (atan(sigma)/M_PI + 0.485)/0.97;
		sigma += randh();
		wrap(sigma, 0., 1.);
		sigma = tan(M_PI*(0.97*sigma - 0.485));
		sigma = exp(sigma);
	}
	else
	{
		nu = log(nu);
		nu += log(1E2)*randh();
		wrap(nu, log(1.), log(1E2));
		nu = exp(nu);
	}


	return logH;
}

double MyDistribution::log_pdf(const std::vector<double>& vec) const
{
	double logp = 0.;

	// Gaussian
	// logp += -log(sigma) - 0.5*pow((vec[0] - mu)/sigma, 2);

	// t
	logp += gsl_sf_lngamma(0.5*(nu + 1.)) - gsl_sf_lngamma(0.5*nu)
			- 0.5*log(nu) - log(sigma)
			- 0.5*(nu + 1.)*
			log(1. + pow(vec[0] - mu, 2)/(sigma*sigma)/nu);
;

	return logp;
}

#include <iostream>
using namespace std;

void MyDistribution::from_uniform(std::vector<double>& vec) const
{
	// Gaussian
	// vec[0] = mu + sigma*gsl_cdf_ugaussian_Pinv(vec[0]);

	// t
	vec[0] = mu + sigma*gsl_cdf_tdist_Pinv(vec[0], nu);
}

void MyDistribution::to_uniform(std::vector<double>& vec) const
{
	// Gaussian
	// vec[0] = gsl_cdf_ugaussian_P((vec[0] - mu)/sigma);

	// t
	vec[0] = gsl_cdf_tdist_P((vec[0] - mu)/sigma, nu);
}

void MyDistribution::print(std::ostream& out) const
{
	out<<mu<<' '<<sigma<<' '<<nu<<' ';
}

