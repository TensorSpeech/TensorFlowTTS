#ifndef TTSFRONTEND_H
#define TTSFRONTEND_H

#include <iostream>
#include <string>
#include <vector>
#include <regex>
#include <stdio.h>

class TTSFrontend
{
public:

    /**
     * Converting text to phoneIDs. 
     * A tmporary method using command to process text in this demo, 
     * which should to be replaced by a pronunciation processing module.
     *@param strCmd Command to call the method of processor.text_to_sequence()
    */
    TTSFrontend(const std::string &mapperJson,
               const std::string &strCmd):
               _mapperJson(mapperJson),
               _strCmd(strCmd)
    {
        std::cout << "TTSFrontend Init" << std::endl;
        std::cout << _mapperJson << std::endl;
        std::cout << _strCmd << std::endl;
    };

    void text2ids(const std::string &text);

    std::vector<int32_t> getPhoneIds() const {return _phonesIds;}
private:

    const std::string _mapperJson;
    const std::string _strCmd;

    std::vector<int32_t> _phonesIds;

    std::string getCmdResult(const std::string &text);
    std::vector<int32_t> strSplit(const std::string &idStr);
};

#endif // TTSFRONTEND_H