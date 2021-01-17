#include "Voice.h"
#include "ext/ZCharScanner.h"



std::vector<int32_t> Voice::PhonemesToID(const std::string & InTxt)
{
	ZStringDelimiter Delim(InTxt);
	Delim.AddDelimiter(" ");

	std::vector<int32_t> VecPhones;
	VecPhones.reserve(Delim.szTokens());

	for (const auto& Pho : Delim.GetTokens()) 
	{
		size_t ArrID = 0;

        if (VoxUtil::FindInVec<std::string>(Pho, Phonemes, ArrID))
            VecPhones.push_back(PhonemeIDs[ArrID]);
		else
            std::cout << "Voice::PhonemesToID() WARNING: Unknown phoneme " << Pho << std::endl;



	}


	return VecPhones;

}

void Voice::ReadPhonemes(const std::string &PhonemePath)
{
    std::ifstream Phone(PhonemePath);

    std::string Line;
    while (std::getline(Phone, Line))
    {
        if (Line.find("\t") == std::string::npos)
            continue;


        ZStringDelimiter Deline(Line);
        Deline.AddDelimiter("\t");

        Phonemes.push_back(Deline[0]);
        PhonemeIDs.push_back(stoi(Deline[1]));




    }

}

void Voice::ReadSpeakers(const std::string &SpeakerPath)
{
    Speakers = GetLinedFile(SpeakerPath);

}

void Voice::ReadEmotions(const std::string &EmotionPath)
{
    Emotions = GetLinedFile(EmotionPath);

}

void Voice::ReadModelInfo(const std::string &ModelInfoPath)
{

    ModelInfo = "";
    std::vector<std::string> MiLines = GetLinedFile(ModelInfoPath);

    for (const std::string& ss : MiLines)
        ModelInfo += ss + "\n";


}

std::vector<std::string> Voice::GetLinedFile(const std::string &Path)
{
    std::vector<std::string> RetLines;
    std::ifstream Fi(Path);

    if (!Fi.good()) // File not exists, ret empty vec
        return RetLines;

    std::string Line;
    while (std::getline(Fi, Line))
    {
        if (Line.size() > 1)
            RetLines.push_back(Line);


    }

    return RetLines;

}

Voice::Voice(const std::string & VoxPath, const std::string &inName, Phonemizer *InPhn)
{
	MelPredictor.Initialize(VoxPath + "/melgen");
	Vocoder.Initialize(VoxPath + "/vocoder");

    if (InPhn)
        Processor.Initialize(InPhn);


    VoxInfo = VoxUtil::ReadModelJSON(VoxPath + "/info.json");
    Name = inName;
    ReadPhonemes(VoxPath + "/phonemes.txt");
    ReadSpeakers(VoxPath + "/speakers.txt");
    ReadEmotions(VoxPath + "/emotions.txt");


    ReadModelInfo(VoxPath + "/info.txt");







}

void Voice::AddPhonemizer(Phonemizer *InPhn)
{
    Processor.Initialize(InPhn);


}


std::vector<float> Voice::Vocalize(const std::string & Prompt, float Speed, int32_t SpeakerID, float Energy, float F0, int32_t EmotionID)
{

    std::string PhoneticTxt = Processor.ProcessTextPhonetic(Prompt,Phonemes,(ETTSLanguage::Enum)VoxInfo.Language);

    TFTensor<float> Mel = MelPredictor.DoInference(PhonemesToID(PhoneticTxt), SpeakerID, Speed, Energy, F0,EmotionID);

	TFTensor<float> AuData = Vocoder.DoInference(Mel);


	int64_t Width = AuData.Shape[0];
	int64_t Height = AuData.Shape[1];
	int64_t Depth = AuData.Shape[2];
	//int z = 0;

	std::vector<float> AudioData;
	AudioData.resize(Height);

	// Code to access 1D array as if it were 3D
	for (int64_t x = 0; x < Width;x++)
	{
		for (int64_t z = 0;z < Depth;z++)
		{
			for (int64_t y = 0; y < Height;y++) {
				int64_t Index = x * Height * Depth + y * Depth + z;
				AudioData[(size_t)y] = AuData.Data[(size_t)Index];

			}

		}
	}


    return AudioData;
}


Voice::~Voice()
{
}
