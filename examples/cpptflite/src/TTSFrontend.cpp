#include "TTSFrontend.h"

void TTSFrontend::text2ids(const std::string &text)
{
    _phonesIds = strSplit(getCmdResult(text));
}

std::string TTSFrontend::getCmdResult(const std::string &text)
{
    char buf[1000] = {0};
    FILE *pf = NULL;

    if( (pf = popen((_strCmd + " " + _mapperJson + " \"" + text + "\"").c_str(), "r")) == NULL )
    {
        return "";
    }

    while(fgets(buf, sizeof(buf), pf))
    {
        continue;
    }

    std::string strResult(buf);
    pclose(pf);

    return strResult;
}

std::vector<int32_t> TTSFrontend::strSplit(const std::string &idStr)
{
    std::vector<int32_t> idsVector;

    std::regex rgx ("\\s+");
    std::sregex_token_iterator iter(idStr.begin(), idStr.end(), rgx, -1);
    std::sregex_token_iterator end;

    while (iter != end)  {
        idsVector.push_back(stoi(*iter));
        ++iter;
    }

    return idsVector;
}