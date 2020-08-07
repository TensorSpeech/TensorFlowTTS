#include "Voice.h"
#include "ext/ZCharScanner.h"
const std::vector<std::string> Phonemes = { "AA","AA0","AA1","AA2","AE","AE0","AE1","AE2","AH","AH0","AH1","AH2","AO","AO0","AO1",
"AO2","AW","AW0","AW1","AW2","AY","AY0","AY1","AY2","B","CH","D","DH","EH","EH0","EH1","EH2","ER","ER0","ER1","ER2","EY","EY0","EY1",
"EY2","F","G","HH","IH","IH0","IH1","IH2","IY","IY0","IY1","IY2","JH","K","L","M","N","NG","OW","OW0","OW1","OW2","OY","OY0","OY1","OY2",
"P","R","S","SH","T","TH","UH","UH0","UH1","UH2","UW","UW0","UW1","UW2","V","W","Y","Z","ZH","SIL","END" };

const std::vector<int32_t> PhonemeIDs = { 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76,
77, 78, 79, 80, 81, 82, 83, 84, 85,  86, 87, 88, 89, 90, 91, 92,93, 94, 95, 96, 97,
98,99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136,
137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149 };


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
			cout << "Voice::PhonemesToID() WARNING: Unknown phoneme " << Pho << endl;



	}
	// Prevent out of range error in single word input
	if (VecPhones.size() > 1)
	{
		if (VecPhones[VecPhones.size() - 1] != 148)
			VecPhones.push_back(148);
	}
	else 
	{
		VecPhones.push_back(148);

	}


	return VecPhones;

}

Voice::Voice(const std::string & VoxPath)
{
	MelPredictor.Initialize(VoxPath + "/melgen");
	Vocoder.Initialize(VoxPath + "/vocoder");
	Processor.Initialize(VoxPath + "/g2p.fst");

}

std::vector<float> Voice::Vocalize(const std::string & Prompt, float Speed, int32_t SpeakerID, float Energy, float F0)
{
	cout << Prompt << endl;

	std::string PhoneticTxt = Processor.ProcessTextPhonetic(Prompt);
	cout << PhoneticTxt << endl;

	TFTensor<float> Mel = MelPredictor.DoInference(PhonemesToID(PhoneticTxt), SpeakerID, Speed, Energy, F0);

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
