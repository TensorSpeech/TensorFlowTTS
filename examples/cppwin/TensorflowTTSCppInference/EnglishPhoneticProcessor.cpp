#include "EnglishPhoneticProcessor.h"

using namespace std;

bool EnglishPhoneticProcessor::Initialize(const std::string & PhoneticModelFn)
{
	if (!FileExists(PhoneticModelFn))
		return false;

	Phonemizer = new PhonetisaurusScript(PhoneticModelFn);



	return true;
}

std::string EnglishPhoneticProcessor::ProcessTextPhonetic(const std::string & InText)
{
	if (!Phonemizer)
		return "ERROR";

	vector<string> Words = Tokenizer.Tokenize(InText);

	string Assemble = "";
	for (size_t w = 0; w < Words.size();w++) 
	{
		const string& Word = Words[w];

		if (Word == "SIL") {
			Assemble.append(Word);
			Assemble.append(" ");


			continue;

		}

		vector<PathData> PhResults = Phonemizer->Phoneticize(Word, 1, 10000, 99.f, false, false, 0.99);
		for (const auto& padat : PhResults) {
			for (const auto& uni : padat.Uniques) {
				Assemble.append(Phonemizer->osyms_->Find(uni));
				Assemble.append(" ");
			}


		}




	}
	

	// Delete last space if there is


	if (Assemble[Assemble.size() - 1] == ' ')
		Assemble.pop_back();


	return Assemble;
}

EnglishPhoneticProcessor::EnglishPhoneticProcessor()
{
	Phonemizer = nullptr;
}

EnglishPhoneticProcessor::EnglishPhoneticProcessor(const std::string & PhModelFn)
{
	Phonemizer = nullptr;
	Initialize(PhModelFn);
}


EnglishPhoneticProcessor::~EnglishPhoneticProcessor()
{
	if (Phonemizer)
		delete Phonemizer;
}
