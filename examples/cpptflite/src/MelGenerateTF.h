#ifndef MELGENERATETF_H
#define MELGENERATETF_H

#include "TfliteBase.h"

using std::vector;

class MelGenerateTF : public TfliteBase
{
public:

    MelGenerateTF(const char* modelFilename):TfliteBase(modelFilename),
                                             inputIndexs(interpreter->inputs()),
                                             ouptIndex(interpreter->outputs()[1]) {};

    MelGenData infer(const vector<int32_t> inputIds);

private:
    vector<int32_t> _speakerId{0};
    vector<float> _speedRatio{1.0};
    vector<float> _f0Ratio{1.0};
    vector<float> _enegyRatio{1.0};

    const vector<int32_t> inputIndexs;
    const int32_t ouptIndex;

};

#endif