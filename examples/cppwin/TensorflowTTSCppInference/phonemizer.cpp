#include "phonemizer.h"
#include <fstream>
#include "ext/ZCharScanner.h"

int32_t GetID(const std::vector<IdStr>& In, const std::string &InStr)
{
    for (const IdStr& It : In)
        if (It.STR == InStr)
            return It.ID;

    return -1;
}

std::string GetSTR(const std::vector<IdStr>& In, int32_t InID)
{
    for (const IdStr& It : In)
        if (It.ID == InID)
            return It.STR;

    return "";

}

std::vector<IdStr> Phonemizer::GetDelimitedFile(const std::string &InFname)
{


    std::ifstream InFile (InFname);

    int32_t CuID;
    std::string Tok;
    std::vector<IdStr> RetVec;


    std::string Line;
    while (std::getline(InFile, Line)) {

        if (Line.find("\t") == std::string::npos)
            continue;


        ZStringDelimiter Deline(Line);
        Deline.AddDelimiter("\t");

        CuID = stoi(Deline[1]);
        Tok = Deline[0];


        RetVec.push_back(IdStr{CuID,Tok});

    }

    return RetVec;


}

void Phonemizer::LoadDictionary(const std::string &InDictFn)
{



    std::ifstream InFile (InDictFn);

    std::string Word;
    std::string Phn;


    if (Dictionary.size())
        Dictionary.clear();




    std::string Line;
    while (std::getline(InFile, Line)) {

        if (Line.find("\t") == std::string::npos)
            continue;


        ZStringDelimiter Deline(Line);
        Deline.AddDelimiter("\t");

        Word = Deline[0];
        Phn = Deline[1];


        Dictionary.push_back(StrStr{Word,Phn});

    }
    // Sort so lookup can be a bit optimized
    std::sort(Dictionary.begin(),Dictionary.end());


}

std::string Phonemizer::DictLookup(const std::string &InWord)
{

    for (size_t w = 0 ; w < Dictionary.size();w++)
    {
        const StrStr& Entr = Dictionary[w];

        if (Entr.Word.size() != InWord.size())
            continue;

        if (Entr.Word == InWord)
            return Entr.Phn;

    }

    return "";

}




Phonemizer::Phonemizer()
{

}

bool Phonemizer::Initialize(const std::string InPath)
{
    // Load indices
    try {
        CharId = GetDelimitedFile(InPath + "/char2id.txt");
        PhnId = GetDelimitedFile(InPath + "/phn2id.txt");

        // Load model
        G2pModel.Initialize(InPath + "/model");

        LoadDictionary(InPath + "/dict.txt");
    }
    catch (...){
        return false;
    }




    return true;


}

std::string Phonemizer::ProcessWord(const std::string &InWord,float Temperature)
{
    // First we try dictionary lookup
    // This is because the g2p model can be unreliable, we only want to use it for novel sentences

    std::string PhnDict = DictLookup(InWord);
    if (!PhnDict.empty())
        return PhnDict;

    std::vector<int32_t> InIndexes;
    InIndexes.reserve(InWord.size());

    // Turn word into indices
    for (const char ch : InWord)
    {
        std::string Single(1,ch);
        int32_t Idx = GetID(CharId,Single);

        if (Idx != -1)
            InIndexes.push_back(Idx);


    }

    TFTensor<int32_t> PhnPrediction = G2pModel.DoInference(InIndexes,Temperature);


    std::string RetStr = "";
    bool FirstIter = true;

    for (int32_t PhnIdx : PhnPrediction.Data)
    {
        std::string PhnTxt = GetSTR(PhnId,PhnIdx);
        if (!PhnTxt.empty())
        {
            if (!FirstIter)
                RetStr.append(" ");

            RetStr.append(PhnTxt);

        }

        FirstIter = false;
    }



    return  RetStr;

}

std::string Phonemizer::GetPhnLanguage() const
{
    return PhnLanguage;
}

void Phonemizer::SetPhnLanguage(const std::string &value)
{

    PhnLanguage = value;
}

std::string Phonemizer::GetGraphemeChars()
{

    std::string RetAllowed = "";
    for (const IdStr& Idx : CharId)
        RetAllowed.append(Idx.STR);

    return RetAllowed;

}




bool operator<(const StrStr &right, const StrStr &left)
{
  return right.Word.length() < left.Word.length();
}
