#ifndef _Data_
#define _Data_

#include <Eigen/Dense>


typedef Eigen::VectorXd Vector;
typedef Eigen::MatrixXd Matrix;

class Data
{
	private:
		

	public:
		Data();
		void load(const char* filename);

	// Singleton
	private:
		static Data instance;
	public:
		static Data& get_instance() { return instance; }
};

#endif

