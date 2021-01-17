#include "tfg2p.h"
#include <stdexcept>
TFG2P::TFG2P()
{
    G2P = nullptr;

}

TFG2P::TFG2P(const std::string &SavedModelFolder)
{
    G2P = nullptr;

    Initialize(SavedModelFolder);
}

bool TFG2P::Initialize(const std::string &SavedModelFolder)
{
    try {

        G2P = new Model(SavedModelFolder);

    }
    catch (...) {
        G2P = nullptr;
        return false;

    }
    return true;
}

TFTensor<int32_t> TFG2P::DoInference(const std::vector<int32_t> &InputIDs, float Temperature)
{
    if (!G2P)
        throw std::invalid_argument("Tried to do inference on unloaded or invalid model!");

    // Convenience reference so that we don't have to constantly derefer pointers.
    Model& Mdl = *G2P;


    // Convenience reference so that we don't have to constantly derefer pointers.

    Tensor input_ids{ Mdl,"serving_default_input_ids" };
    Tensor input_len{Mdl,"serving_default_input_len"};
    Tensor input_temp{Mdl,"serving_default_input_temperature"};

    input_ids.set_data(InputIDs, std::vector<int64_t>{(int64_t)InputIDs.size()});
    input_len.set_data(std::vector<int32_t>{(int32_t)InputIDs.size()});
    input_temp.set_data(std::vector<float>{Temperature});



    std::vector<Tensor*> Inputs {&input_ids,&input_len,&input_temp};
    Tensor out_ids{ Mdl,"StatefulPartitionedCall" };

    Mdl.run(Inputs, out_ids);

    TFTensor<int32_t> RetTensor = VoxUtil::CopyTensor<int32_t>(out_ids);

    return RetTensor;


}

TFG2P::~TFG2P()
{
    if (G2P)
        delete G2P;

}
