#ifndef PHONEMIZER_H
#define PHONEMIZER_H
#include "tfg2p.h"
#include <tuple>
#include <set>
#include <algorithm>

struct IdStr{
    int32_t ID;
    std::string STR;
};


struct StrStr{
    std::string Word;
    std::string Phn;
};


class Phonemizer
{
private:
    TFG2P G2pModel;

    std::vector<IdStr> CharId;
    std::vector<IdStr> PhnId;






    std::vector<IdStr> GetDelimitedFile(const std::string& InFname);


    // Sorry, can't use set, unordered_map or any other types. (I tried)
    std::vector<StrStr> Dictionary;

    void LoadDictionary(const std::string& InDictFn);

    std::string DictLookup(const std::string& InWord);



    std::string PhnLanguage;
public:
    Phonemizer();
    /*
     * Initialize a phonemizer
     * Expects:
     * - Two files consisting in TOKEN \t ID:
     * -- char2id.txt: Translation from input character to ID the model can accept
     * -- phn2id.txt: Translation from output ID from the model to phoneme
     * - A model/ folder where a G2P-Tensorflow model was saved as SavedModel
     * - dict.txt: Phonetic dictionary. First it searches the word there and if it can't be found then it uses the model.

    */
    bool Initialize(const std::string InPath);
    std::string ProcessWord(const std::string& InWord, float Temperature = 0.1f);
    std::string GetPhnLanguage() const;
    void SetPhnLanguage(const std::string &value);

    std::string GetGraphemeChars();

};


bool operator<(const StrStr& right,const StrStr& left);
#endif // PHONEMIZER_H
