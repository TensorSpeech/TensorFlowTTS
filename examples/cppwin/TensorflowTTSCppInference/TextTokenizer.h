#pragma once
#include <vector>
#include <string>

class TextTokenizer
{
private:
	std::string IntToStr(int number);

	std::vector<std::string> ExpandNumbers(const std::vector<std::string>& SpaceTokens);
public:
	TextTokenizer();
	~TextTokenizer();

	std::vector<std::string> Tokenize(const std::string& InTxt);
};

