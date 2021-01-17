#pragma once

#include "ext/CppFlow/include/Model.h"
#include "VoxCommon.hpp"
class FastSpeech2
{
private:
	Model* FastSpeech;

public:
	FastSpeech2();
	FastSpeech2(const std::string& SavedModelFolder);

	/*
	Initialize and load the model
	
	-> SavedModelFolder: Folder where the .pb, variables, and other characteristics of the exported SavedModel
	<- Returns: (bool)Success 
	*/
	bool Initialize(const std::string& SavedModelFolder);

	/*
	Do inference on a FastSpeech2 model.

	-> InputIDs: Input IDs of tokens for inference
	-> SpeakerID: ID of the speaker in the model to do inference on. If single speaker, always leave at 0. If multispeaker, refer to your model.
	-> Speed, Energy, F0: Parameters for FS2 inference. Leave at 1.f for defaults

	<- Returns: TFTensor<float> with shape {1,<len of mel in frames>,80} containing contents of mel spectrogram. 
	*/
    TFTensor<float> DoInference(const std::vector<int32_t>& InputIDs, int32_t SpeakerID = 0, float Speed = 1.f, float Energy = 1.f, float F0 = 1.f,int32_t EmotionID = -1);



	~FastSpeech2();
};

