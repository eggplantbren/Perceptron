#include "MyModel.h"

MyModel::MyModel()
:weights(1,
         Data::get_instance().get_dim_outputs()*
                Data::get_instance().get_dim_inputs(),
         true,
         MyConditionalPrior())
{

}

