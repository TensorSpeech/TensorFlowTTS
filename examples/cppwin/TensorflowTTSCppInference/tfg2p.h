#ifndef TFG2P_H
#define TFG2P_H
#include "ext/CppFlow/include/Model.h"
#include "VoxCommon.hpp"


class TFG2P
{
private:
    Model* G2P;

public:
    TFG2P();
    TFG2P(const std::string& SavedModelFolder);

    /*
    Initialize and load the model

    -> SavedModelFolder: Folder where the .pb, variables, and other characteristics of the exported SavedModel
    <- Returns: (bool)Success
    */
    bool Initialize(const std::string& SavedModelFolder);

    /*
    Do inference on a G2P-TF-RNN model.

    -> InputIDs: Input IDs of tokens for inference
    -> Temperature: Temperature of the RNN, values higher than 0.1 cause instability.

    <- Returns: TFTensor<int32_t> containing phoneme IDs
    */
    TFTensor<int32_t> DoInference(const std::vector<int32_t>& InputIDs, float Temperature = 0.1f);

    ~TFG2P();

};

#endif // TFG2P_H
