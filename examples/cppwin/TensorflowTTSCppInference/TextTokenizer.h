#pragma once
#include <vector>
#include <string>
#include "VoxCommon.hpp"

class TextTokenizer
{
private:
    std::string AllowedChars;
	std::string IntToStr(int number);

	std::vector<std::string> ExpandNumbers(const std::vector<std::string>& SpaceTokens);
public:
	TextTokenizer();
	~TextTokenizer();

    std::vector<std::string> Tokenize(const std::string& InTxt,ETTSLanguage::Enum Language = ETTSLanguage::English);
    void SetAllowedChars(const std::string &value);
};

