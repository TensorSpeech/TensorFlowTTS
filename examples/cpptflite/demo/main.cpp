#include <iostream>
#include <iterator>
#include <vector>
#include <sys/time.h>
#include "MelGenerateTF.h"
#include "VocoderTF.h"
#include "VoxCommon.h"

int main(int argc, char* argv[])
{
    if (argc != 3) 
    {
        fprintf(stderr, "minimal <tflite models>\n");
        return 1;
    }
    const char* melgenfile  = argv[1];
    const char* vocoderfile = argv[2];

    // std::vector<int32_t> input_ids{1,16,78,11,48,3,4,193,21,114,23,98,8,27,3,12,116,22,86,21,114,4,74,14,122,4,74,14,122,16,87,21,112,4,74,4,209,21,112,19,114,1,286};
    std::vector<int32_t> input_ids{1, 27, 56, 2, 23, 116, 2, 6, 79, 2, 12, 56, 2, 15, 33, 2, 6, 204, 2, 10, 57, 2, 10, 168, 2, 10, 51, 2, 10,
                                    168, 2, 27, 143, 2, 6, 184, 2, 6, 200, 2, 6, 118, 2, 13, 54, 2, 9, 69, 2, 25, 81, 2, 24, 145, 1, 218};

    MelGenerateTF MelGen(melgenfile);
    VocoderTF Vocoder(vocoderfile);

    MelGenData mel = MelGen.infer(input_ids);
    vector<float> audio = Vocoder.infer(mel);

#if 1
    std::cout << "********* MEL SHAPE **********" << std::endl;
    for (auto index : mel.melShape)
    {
        std::cout << index << " ";
    }
    std::cout << "\n" << std::endl;
    std::cout << "********* AUDIO LEN **********" << std::endl;
    std::cout << audio.size() << std::endl;

    VoxUtil::ExportWAV("1.wav", audio, 24000);
    std::cout << "Wavfile: 1.wav creats." << std::endl;

/* Make sure NOT ERROR by infer repeatly */
    // std::vector<int32_t> input_ids2{1,4,76,4,74,20,36,20,109,4,153,14,134,18,76,5,26,12,133,21,112,1,286};

    // MelGenData mel2 = MelGen.infer(input_ids2);

    // std::cout << "********* MEL SHAPE **********" << std::endl;
    // for (auto index : mel2.melShape)
    // {
    //     std::cout << index << " ";
    // }
    // std::cout << "\n" << std::endl;

    // vector<float> audio2 = Vocoder.infer(mel2);

    // std::cout << "********* AUDIO LEN **********" << std::endl;
    // std::cout << audio2.size() << std::endl;

    // VoxUtil::ExportWAV("2.wav", audio2, 24000);

#else
    struct timeval sTime, eTime;
    long long alltime = 0;
    long long melgentime = 0;
    long long vocodertime = 0;
    long long exeTime;

    int32_t n = 10;
    int32_t nn = n;
    while (n--)
    {   
        gettimeofday(&sTime, NULL);

        MelGenData mel = MelGen.infer(input_ids);

        gettimeofday(&eTime, NULL);

        exeTime = (eTime.tv_sec-sTime.tv_sec)*1000000+(eTime.tv_usec-sTime.tv_usec);
        alltime += exeTime;
        melgentime += exeTime;

        std::cout << n << " : melgen:" << exeTime / 1000 << " || ";

        gettimeofday(&sTime, NULL);

        vector<float> audio = Vocoder.infer(mel);

        gettimeofday(&eTime, NULL);
        exeTime = (eTime.tv_sec-sTime.tv_sec)*1000000+(eTime.tv_usec-sTime.tv_usec);
        alltime += exeTime;
        vocodertime += exeTime;

        std::cout << "vocoder:" << exeTime / 1000 << std::endl;
    }

    std::cout << "average- melgen: " << melgentime / nn / 1000 << "; vocoder: " << vocodertime / nn / 1000 << std::endl;
    std::cout << "average- " << alltime / nn / 1000 << std::endl;
#endif

}