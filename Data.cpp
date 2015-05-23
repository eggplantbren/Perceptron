#include "Data.h"
#include <fstream>
#include <iostream>

using namespace std;

Data Data::instance;

Data::Data()
{

}

void Data::load(const char* filename)
{
	fstream fin(filename, ios::in);
	if(!fin)
	{
		cerr<<"# Failed to load file "<<filename<<"."<<endl;
		exit(0);
	}

	char junk;
	fin>>junk;
	fin>>dim_inputs;
	fin>>dim_outputs;
	inputs.clear();
	outputs.clear();

	Vector input(dim_inputs);
	Vector output(dim_outputs);

	bool success = true;
	while(success)
	{
		success = true;
		for(int i=0; i<dim_inputs; i++)
			success = (success) && (fin>>input(i));
		for(int i=0; i<dim_outputs; i++)
			success = (success) && (fin>>output(i));

		if(success)
		{
			inputs.push_back(input);
			outputs.push_back(output);
		}
	}
	fin.close();

	cout<<"# Loaded "<<inputs.size()<<" examples from file "<<filename;
	cout<<"."<<endl;
}

