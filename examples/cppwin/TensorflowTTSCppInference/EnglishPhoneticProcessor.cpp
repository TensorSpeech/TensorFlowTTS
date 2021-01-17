#include "EnglishPhoneticProcessor.h"
#include "VoxCommon.hpp"

using namespace std;

bool EnglishPhoneticProcessor::Initialize(Phonemizer* InPhn)
{


    Phoner = InPhn;
    Tokenizer.SetAllowedChars(Phoner->GetGraphemeChars());



	return true;
}

std::string EnglishPhoneticProcessor::ProcessTextPhonetic(const std::string& InText, const std::vector<string> &InPhonemes,ETTSLanguage::Enum InLanguage)
{
    if (!Phoner)
		return "ERROR";



    vector<string> Words = Tokenizer.Tokenize(InText,InLanguage);

	string Assemble = "";
    // Make a copy of the dict passed.

	for (size_t w = 0; w < Words.size();w++) 
	{
		const string& Word = Words[w];

        if (Word.find("@") != std::string::npos){
            std::string AddPh = Word.substr(1); // Remove the @
            size_t OutId = 0;
            if (VoxUtil::FindInVec(AddPh,InPhonemes,OutId))
            {
                Assemble.append(InPhonemes[OutId]);
                Assemble.append(" ");


            }

            continue;

        }




        size_t OverrideIdx = 0;



        std::string Res = Phoner->ProcessWord(Word,0.001f);

        // Cache the word in the override dict so next time we don't have to research it

        Assemble.append(Res);
        Assemble.append(" ");





	}
	

	// Delete last space if there is


	if (Assemble[Assemble.size() - 1] == ' ')
		Assemble.pop_back();


	return Assemble;
}

EnglishPhoneticProcessor::EnglishPhoneticProcessor()
{
    Phoner = nullptr;
}

EnglishPhoneticProcessor::EnglishPhoneticProcessor(Phonemizer *InPhn)
{
    Initialize(InPhn);

}



EnglishPhoneticProcessor::~EnglishPhoneticProcessor()
{
    if (Phoner)
        delete Phoner;
}
