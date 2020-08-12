#include "ZCharScanner.h"
#include <stdexcept>
using namespace std;

int ZStringDelimiter::key_search(const GString& s, const GString& key)
{
	int count = 0;
	size_t pos = 0;
	while ((pos = s.find(key, pos)) != GString::npos) {
		++count;
		++pos;
	}
	return count;
}
void ZStringDelimiter::UpdateTokens()
{
	if (!m_vDelimiters.size() || m_sString == "")
		return;

	m_vTokens.clear();


	vector<GString>::iterator dIt = m_vDelimiters.begin();
	while (dIt != m_vDelimiters.end())
	{
		GString delimiter = *dIt;
	

		DelimStr(m_sString, delimiter, true);
		
	
		++dIt;
	}
	
	

}


void ZStringDelimiter::DelimStr(const GString & s, const GString & delimiter, const bool & removeEmptyEntries)
{
	BarRange(0, s.length());
	for (size_t start = 0, end; start < s.length(); start = end + delimiter.length())
	{
		size_t position = s.find(delimiter, start);
		end = position != GString::npos ? position : s.length();

		GString token = s.substr(start, end - start);
		if (!removeEmptyEntries || !token.empty())
		{
			if (token != s)
				m_vTokens.push_back(token);

		}
		Bar(position);
	}

	// dadwwdawdaawdwadwd
}

void ZStringDelimiter::BarRange(const int & min, const int & max)
{
#ifdef _AFX_ALL_WARNINGS
	if (PgBar)
		m_pBar->SetRange32(min, max);


#endif
}

void ZStringDelimiter::Bar(const int & pos)
{
#ifdef _AFX_ALL_WARNINGS
	if (PgBar)
		m_pBar->SetPos(pos);


#endif
}

ZStringDelimiter::ZStringDelimiter()
{
	m_sString = "";
	tokenIndex = 0;
	PgBar = false;
}


bool ZStringDelimiter::GetFirstToken(GString & in_out)
{
	if (m_vTokens.size() >= 1) {
		in_out = m_vTokens[0];
		return true;
	}
	else {
		return false;
	}
}

bool ZStringDelimiter::GetNextToken(GString & in_sOut)
{
	if (tokenIndex > m_vTokens.size() - 1)
		return false;

	in_sOut = m_vTokens[tokenIndex];
	++tokenIndex;

	return true;
}

GString ZStringDelimiter::operator[](const size_t & in_index)
{
	if (in_index > m_vTokens.size())
		throw std::out_of_range("ZStringDelimiter tried to access token higher than size");

	return m_vTokens[in_index];

}
GString ZStringDelimiter::Reassemble(const GString& delim, const int& nelem)
{
	GString Result = "";
	TokenIterator RasIt = m_vTokens.begin();
	int r = 0;
	if (nelem == -1) {
		while (RasIt != m_vTokens.end())
		{

			if (r != 0)
				Result.append(delim);

			Result.append(*RasIt);

			++r;


			++RasIt;
		}
	}
	else {
		while (RasIt != m_vTokens.end() && r < nelem)
		{
		
			if (r != 0)
				Result.append(delim);

			Result.append(*RasIt);

			++r;
			++RasIt;
		}
	}
	
	return Result;

}

GString ZStringDelimiter::Reassemble(const GString & delim, const std::vector<GString>& Strs,int nelem)
{
	GString Result = "";
	TokenIterator RasIt = Strs.begin();
	int r = 0;
	if (nelem == -1) {
		while (RasIt != Strs.end())
		{

			if (r != 0)
				Result.append(delim);

			Result.append(*RasIt);

			++r;


			++RasIt;
		}
	}
	else {
		while (RasIt != Strs.end() && r < nelem)
		{

			if (r != 0)
				Result.append(delim);

			Result.append(*RasIt);

			++r;
			++RasIt;
		}
	}

	return Result;
}

void ZStringDelimiter::AddDelimiter(const GString & in_Delim)
{
	m_vDelimiters.push_back(in_Delim);
	UpdateTokens();

}

ZStringDelimiter::~ZStringDelimiter()
{
}
