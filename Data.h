#ifndef _Data_
#define _Data_

#include <vector>
#include <Eigen/Dense>

typedef Eigen::VectorXd Vector;
typedef Eigen::MatrixXd Matrix;

class Data
{
	private:
		int dim_inputs, dim_outputs;

		std::vector<Vector> inputs;
		std::vector<Vector> outputs;

	public:
		Data();
		void load(const char* filename);

		// Getters
		const std::vector<Vector>& get_inputs() const
		{ return inputs; }
		const std::vector<Vector>& get_outputs() const
		{ return outputs; }

	// Singleton
	private:
		static Data instance;
	public:
		static Data& get_instance() { return instance; }
};

#endif

