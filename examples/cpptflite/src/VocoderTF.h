#ifndef VOCODERTF_H
#define VOCODERTF_H

#include "TfliteBase.h"

using std::vector;

class VocoderTF : public TfliteBase
{
public:

    VocoderTF(const char* modelFilename):TfliteBase(modelFilename),
                                         inputIndex(interpreter->inputs()[0]),
                                         outputIndex(interpreter->outputs()[0]) {};

    vector<float> infer(const MelGenData mel);

private:

    const int32_t inputIndex;
    const int32_t outputIndex;
};

#endif