#pragma once

#include "ext/CppFlow/include/Model.h"
#include "VoxCommon.hpp"
class MultiBandMelGAN
{
private:
	Model* MelGAN;


public:
	bool Initialize(const std::string& VocoderPath);


	// Do MultiBand MelGAN inference including PQMF
	// -> InMel:  Mel spectrogram (shape [1, xx, 80])
	// <- Returns: Tensor data [4, xx, 1]
	TFTensor<float> DoInference(const TFTensor<float>& InMel);

	MultiBandMelGAN();
	~MultiBandMelGAN();
};

