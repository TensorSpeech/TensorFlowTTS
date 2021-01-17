
#include <iostream>
#include "Voice.h"
#define LOGF(txt) std::cout << txt <<  "\n"
#include "phonemizer.h"
#include "ext/ZCharScanner.h"
#include <algorithm>
#include <cctype>
#include <string>
#include "ext/cxxopts.hpp"

std::vector<std::string> GetTxtFile(const std::string& InFn) {
 
	std::vector<std::string> Ret;
    std::ifstream InFile(InFn);

	if (!InFile.good())
		return Ret;


    std::string Line;
    while (std::getline(InFile, Line)) 
    {
		Ret.push_back(Line);



    }
	InFile.close();

	return Ret;
 


    
}

std::vector<std::string> SuperWordSplit(const std::string& InStr, int MaxLen)
{
	ZStringDelimiter Del1(InStr);
	Del1.AddDelimiter(" ");
	
	std::vector<std::string> RawWords = Del1.GetTokens();
	int AmtWords = RawWords.size();

	int Idx = 0;
	std::string CurrentStr = "";

	std::vector<std::string> SplitStrs;

	while (Idx < AmtWords)
	{
		if (CurrentStr.size() > 0)
			CurrentStr.append(" ");

		std::string CuWord = RawWords[Idx];
		// phonetic input has to be uppercase
		if (CuWord.find("@") == std::string::npos) 
		{
			std::transform(CuWord.begin(), CuWord.end(), CuWord.begin(),
				[](unsigned char c) { return std::tolower(c); });
		} 


		CurrentStr.append(CuWord);

		if (CurrentStr.length() > MaxLen) {
			SplitStrs.push_back(CurrentStr);
			CurrentStr = "";

		}


		Idx += 1;

		// Add the last string
		if (Idx == AmtWords)
			SplitStrs.push_back(CurrentStr);






	}

	return SplitStrs;

}

int main(int argc, char* argv[])
{
	cxxopts::Options options("TFTTSInfer", "Inference with TensorflowTTS models in command line");
	options.add_options()
		("v,voice", "Path to the voice folder", cxxopts::value<std::string>()->default_value("LJ")) // a bool parameter
		("l,language", "Path to the language folder for G2P", cxxopts::value<std::string>()->default_value("g2p/English"))
		("o,output", "Name of .wav file output of all infers", cxxopts::value<std::string>()->default_value("AllAud.wav"))
		("m,maxlen", "Optional, max length of split for TTS. Default is 180", cxxopts::value<int>()->default_value("180"))
		;

	auto Args = options.parse(argc, argv);

	std::string Name = Args["voice"].as<std::string>();
	std::string Lang = Args["language"].as<std::string>();
	std::string OutputFileName = Args["output"].as<std::string>();
	int MaxLen = Args["maxlen"].as<int>();

	if (OutputFileName.find(".wav") == std::string::npos)
		OutputFileName += ".wav";



	LOGF("Loading voice...");

	// Load phonemizer
	Phonemizer StdPh;

	bool G2pInit = StdPh.Initialize(Lang);
	if (!G2pInit) {
		LOGF("Could not initialize language and/or G2P model! See if the path is correct and try again!");
		return -2;
	
	}

	// Load the voice itself
	Voice CurrentVox(Name,Name,&StdPh);
	std::vector<float> AllAud;

	// Begin interactive console
	bool Running = true;
	while (Running)
	{
		std::string Prompt = "";

		LOGF("Type a prompt, or type EXIT to exit ");

		std::getline(std::cin, Prompt);
		if (Prompt == "EXIT") {
			Running = false;
			break;
		}
		std::vector<float> Audata;

		// Split the prompt into chunks (if the user inputs like that)
		for (const auto& Spli : SuperWordSplit(Prompt, MaxLen)) {
			std::vector<float> ImmediateAudata = CurrentVox.Vocalize(Prompt + CurrentVox.GetInfo().EndPadding);
			// Insert the audio data to the end of the mid-level audata vector
			Audata.insert(Audata.end(), ImmediateAudata.begin(), ImmediateAudata.end());

		
		}




		std::string Filename = Prompt.substr(0, std::min(16, (int)Prompt.size())) + ".wav";

		VoxUtil::ExportWAV(Filename, Audata, CurrentVox.GetInfo().SampleRate);

		// Insert the audio into the AllAud vector
		AllAud.insert(AllAud.end(), Audata.begin(), Audata.end());
		
		LOGF("Saved to " + Filename);




	}


	// Export all the audio
	VoxUtil::ExportWAV(OutputFileName, AllAud, CurrentVox.GetInfo().SampleRate);
	LOGF("Saved ALL to " + OutputFileName);

	std::cout << "Hello TensorflowTTS!\n";
	return 0;

}
