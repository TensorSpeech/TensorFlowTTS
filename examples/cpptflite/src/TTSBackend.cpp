#include "TTSBackend.h"

void TTSBackend::inference(std::vector<int32_t> phonesIds)
{
    _mel = MelGen.infer(phonesIds);
    _audio = Vocoder.infer(_mel);
}