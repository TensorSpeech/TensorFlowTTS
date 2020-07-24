#!/bin/bash
mkdir mfa
cd mfa
wget https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/releases/download/v1.1.0-beta.2/montreal-forced-aligner_linux.tar.gz
tar -zxvf montreal-forced-aligner_linux.tar.gz
cd mfa
mkdir lexicon
cd lexicon
wget http://www.openslr.org/resources/11/librispeech-lexicon.txt