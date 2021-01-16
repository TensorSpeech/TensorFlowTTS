#include <iostream>
#include <vector>
#include <string>
#include <regex>
#include <stdio.h>

using namespace std;

string getCmdResult(const string &strCmd)
{
    char buf[1000] = {0};
    FILE *pf = NULL;

    if( (pf = popen(strCmd.c_str(), "r")) == NULL )
    {
        return "";
    }

    while(fgets(buf, sizeof(buf), pf))
    {
        continue;
    }

    string strResult(buf);
    pclose(pf);

    return strResult;
}

vector<int32_t> strSplit(const string &text)
{
    vector<int32_t> ids;

    regex rgx ("\\s+");
    sregex_token_iterator iter(text.begin(), text.end(), rgx, -1);
    sregex_token_iterator end;

    while (iter != end)  {
        ids.push_back(stoi(*iter));
        ++iter;
    }

    return ids;
}

int main(int argc, char* argv[])
{
    if (argc != 2) 
    {
        fprintf(stderr, "demo hanzi\n");
        return 1;
    }

    string cmd = "python3 text2ids.py " + string(argv[1]);

    string tmp = getCmdResult(cmd);

    vector<int32_t> ids = strSplit(tmp);

    cout << "******************" << endl;

    for (auto iter: ids)
    {
        cout << iter << " ";
    }
    cout << endl;

    return 0;
}
