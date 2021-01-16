#ifndef TTSBACKEND_H
#define TTSBACKEND_H

#include <iostream>
#include <vector>
#include "MelGenerateTF.h"
#include "VocoderTF.h"

class TTSBackend
{
public:
    TTSBackend(const char* melgenfile, const char* vocoderfile):
               MelGen(melgenfile), Vocoder(vocoderfile) 
    {
        std::cout << "TTSBackend Init" << std::endl;
        std::cout << melgenfile << std::endl;
        std::cout << vocoderfile << std::endl;
    };

    void inference(std::vector<int32_t> phonesIds);

    MelGenData getMel() const {return _mel;}
    std::vector<float> getAudio() const {return _audio;}

private:
    MelGenerateTF MelGen;
    VocoderTF Vocoder;

    MelGenData _mel;
    std::vector<float> _audio;
};

#endif // TTSBACKEND_H