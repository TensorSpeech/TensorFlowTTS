#pragma once
#include "TextTokenizer.h"

#ifndef PHONETISAURUS_INCLUDED
#define PHONETISAURUS_INCLUDED
// Find a way to stop the linker from spitting:
// "int __cdecl LoadClusters(class fst::SymbolTable const *,class std::unordered_map<int,class std::vector<int,class std::allocator<int> >,struct std::hash<int>,struct std::equal_to<int>,class std::allocator<struct std::pair<int const ,class std::vector<int,class std::allocator<int> > > > > *,class std::unordered_map<class std::vector<int,class std::allocator<int> >,int,struct VectorIntHash,struct std::equal_to<class std::vector<int,class std::allocator<int> > >,class std::allocator<struct std::pair<class std::vector<int,class std::allocator<int> > const ,int> > > *)" (?LoadClusters@@YAHPEBVSymbolTable@fst@@PEAV?$unordered_map@HV?$vector@HV?$allocator@H@std@@@std@@U?$hash@H@2@U?$equal_to@H@2@V?$allocator@U?$pair@$$CBHV?$vector@HV?$allocator@H@std@@@std@@@std@@@2@@std@@PEAV?$unordered_map@V?$vector@HV?$allocator@H@std@@@std@@HUVectorIntHash@@U?$equal_to@V?$vector@HV?$allocator@H@std@@@std@@@2@V?$allocator@U?$pair@$$CBV?$vector@HV?$allocator@H@std@@@std@@H@std@@@2@@4@@Z) already defined in EnglishPhoneticProcessor.obj
#include <include/PhonetisaurusScript.h>
#endif
class EnglishPhoneticProcessor
{
private:
	TextTokenizer Tokenizer;
	PhonetisaurusScript* Phonemizer;

	inline bool FileExists(const std::string& name) {
		ifstream f(name.c_str());
		return f.good();
	}

public:
	bool Initialize(const std::string& PhoneticModelFn);
	std::string ProcessTextPhonetic(const std::string& InText);
	EnglishPhoneticProcessor();
	EnglishPhoneticProcessor(const std::string& PhModelFn);
	~EnglishPhoneticProcessor();
};

