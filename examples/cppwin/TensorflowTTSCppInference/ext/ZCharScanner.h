#pragma once

#define GBasicCharScanner ZStringDelimiter

#include <vector>
#include <string>

#define ZSDEL_USE_STD_STRING
#ifndef ZSDEL_USE_STD_STRING
#include "golem_string.h"
#else
#define GString std::string
#endif

typedef std::vector<GString>::const_iterator TokenIterator;

// ZStringDelimiter
// ==============
// Simple class to delimit and split strings.
// You can use operator[] to access them
// Or you can use the itBegin() and itEnd() to get some iterators
// =================
class ZStringDelimiter
{
private:
	int key_search(const GString & s, const GString & key);
	void UpdateTokens();
	std::vector<GString> m_vTokens;
	std::vector<GString> m_vDelimiters;

	GString m_sString;

	void DelimStr(const GString& s, const GString& delimiter, const bool& removeEmptyEntries = false);
	void BarRange(const int& min, const int& max);
	void Bar(const int& pos);
	size_t tokenIndex;
public:
	ZStringDelimiter();
	bool PgBar;

#ifdef _AFX_ALL_WARNINGS
	CProgressCtrl* m_pBar;
#endif

	ZStringDelimiter(const GString& in_iStr) {
		m_sString = in_iStr;
		PgBar = false;

	}

	bool GetFirstToken(GString& in_out);
	bool GetNextToken(GString& in_sOut);

	// std::String alts

	size_t szTokens() { return m_vTokens.size(); }
	GString operator[](const size_t& in_index);

	GString Reassemble(const GString & delim, const int & nelem = -1);

	// Override to reassemble provided tokens.
	GString Reassemble(const GString & delim, const std::vector<GString>& Strs,int nelem = -1);

	// Get a const reference to the tokens
	const std::vector<GString>& GetTokens() { return m_vTokens; }

	TokenIterator itBegin() { return m_vTokens.begin(); }
	TokenIterator itEnd() { return m_vTokens.end(); }

	void SetText(const GString& in_Txt) { 
		m_sString = in_Txt; 
		if (m_vDelimiters.size())
			UpdateTokens();
	}
	void AddDelimiter(const GString& in_Delim);

	~ZStringDelimiter();
};

