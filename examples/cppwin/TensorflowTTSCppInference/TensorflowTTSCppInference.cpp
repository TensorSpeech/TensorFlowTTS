
#include <iostream>
#include "Voice.h"
#define LOGF(txt) std::cout << txt <<  "\n"

int main()
{
	bool Running = true;
	LOGF("Loading voice...");
	Voice LJSpeech("LJ");
	while (Running) 
	{
		std::string Prompt = "";

		LOGF("Type a prompt, or type EXIT to exit ");

		getline(cin, Prompt);
		if (Prompt == "EXIT") {
			Running = false;
			break;
		}
		std::vector<float> Audata = LJSpeech.Vocalize(Prompt);
		

		std::string Filename = Prompt.substr(0, std::min(16, (int)Prompt.size())) + ".wav";

		VoxUtil::ExportWAV(Filename, Audata, 22050);
		LOGF("Saved to " + Filename);
		


	}


	std::cout << "Hello TensorflowTTS!\n";
	return 0;

}
