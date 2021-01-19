#include <iostream>
#include <iterator>
#include <vector>
#include "VoxCommon.h"
#include "TTSFrontend.h"
#include "TTSBackend.h"

typedef struct 
{
    const char* mapperJson;
    unsigned int sampleRate;
} Processor;

int main(int argc, char* argv[])
{
    if (argc != 3) 
    {
        fprintf(stderr, "demo text wavfile\n");
        return 1;
    }

    const char* cmd        = "python3 ../demo/text2ids.py";

    Processor proc;
#if LJSPEECH
    proc.mapperJson = "../../../tensorflow_tts/processor/pretrained/ljspeech_mapper.json";
    proc.sampleRate = 22050;
#elif BAKER
    proc.mapperJson = "../../../tensorflow_tts/processor/pretrained/baker_mapper.json";
    proc.sampleRate = 24000;
#endif

    const char* melgenfile  = "../models/fastspeech2_quan.tflite";
    const char* vocoderfile = "../models/mb_melgan.tflite";

    // Init
    TTSFrontend ttsfrontend(proc.mapperJson, cmd);
    TTSBackend ttsbackend(melgenfile, vocoderfile);

    // Process
    ttsfrontend.text2ids(argv[1]);
    std::vector<int32_t> phonesIds = ttsfrontend.getPhoneIds();

    ttsbackend.inference(phonesIds);
    MelGenData mel = ttsbackend.getMel();
    std::vector<float> audio = ttsbackend.getAudio();

    std::cout << "********* Phones' ID *********" << std::endl;

    for (auto iter: phonesIds)
    {
        std::cout << iter << " ";
    }
    std::cout << std::endl;

    std::cout << "********* MEL SHAPE **********" << std::endl;
    for (auto index : mel.melShape)
    {
        std::cout << index << " ";
    }
    std::cout << std::endl;

    std::cout << "********* AUDIO LEN **********" << std::endl;
    std::cout << audio.size() << std::endl;

    VoxUtil::ExportWAV(argv[2], audio, proc.sampleRate);
    std::cout << "Wavfile: " << argv[2] << " creats." << std::endl;

    return 0;
}