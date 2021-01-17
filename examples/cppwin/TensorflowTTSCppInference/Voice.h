#pragma once

#include "FastSpeech2.h"
#include "MultiBandMelGAN.h"
#include "EnglishPhoneticProcessor.h"


class Voice
{
private:
	FastSpeech2 MelPredictor;
	MultiBandMelGAN Vocoder;
	EnglishPhoneticProcessor Processor;
    VoiceInfo VoxInfo;



    std::vector<std::string> Phonemes;
    std::vector<int32_t> PhonemeIDs;



	std::vector<int32_t> PhonemesToID(const std::string& InTxt);

    std::vector<std::string> Speakers;
    std::vector<std::string> Emotions;

    void ReadPhonemes(const std::string& PhonemePath);

    void ReadSpeakers(const std::string& SpeakerPath);

    void ReadEmotions(const std::string& EmotionPath);


    void ReadModelInfo(const std::string& ModelInfoPath);

    std::vector<std::string> GetLinedFile(const std::string& Path);


    std::string ModelInfo;

public:
	/* Voice constructor, arguments obligatory.
	 -> VoxPath: Path of folder where models are contained. 
	 --  Must be a folder without an ending slash with UNIX slashes, can be relative or absolute (eg: MyVoices/Karen)
	 --  The folder must contain the following elements:
	 ---  melgen: Folder generated where a FastSpeech2 model was saved as SavedModel, with .pb, variables, etc
	 ---  vocoder: Folder where a Multi-Band MelGAN model was saved as SavedModel.
     ---  info.json: Model information
     ---  phonemes.txt: Tab delimited file containing PHONEME \t ID, for inputting to the FS2 model.

     --- If multispeaker, a lined .txt file called speakers.txt
     --- If multi-emotion, a lined .txt file called emotions.txt

	*/
    Voice(const std::string& VoxPath, const std::string& inName,Phonemizer* InPhn);

    void AddPhonemizer(Phonemizer* InPhn);


    std::vector<float> Vocalize(const std::string& Prompt, float Speed = 1.f, int32_t SpeakerID = 0, float Energy = 1.f, float F0 = 1.f,int32_t EmotionID = -1);

    std::string Name;
    inline const VoiceInfo& GetInfo(){return VoxInfo;}

    inline const std::vector<std::string>& GetSpeakers(){return Speakers;}
    inline const std::vector<std::string>& GetEmotions(){return Emotions;}

    inline const std::string& GetModelInfo(){return ModelInfo;}

	~Voice();
};

