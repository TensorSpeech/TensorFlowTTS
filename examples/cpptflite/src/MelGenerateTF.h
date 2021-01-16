#ifndef MELGENERATETF_H
#define MELGENERATETF_H

#include "TfliteBase.h"

class MelGenerateTF : public TfliteBase
{
public:

    MelGenerateTF(const char* modelFilename):TfliteBase(modelFilename),
                                             inputIndexs(interpreter->inputs()),
                                             ouptIndex(interpreter->outputs()[1]) {};

    MelGenData infer(const std::vector<int32_t> inputIds);

private:
    std::vector<int32_t> _speakerId{0};
    std::vector<float> _speedRatio{1.0};
    std::vector<float> _f0Ratio{1.0};
    std::vector<float> _enegyRatio{1.0};

    const std::vector<int32_t> inputIndexs;
    const int32_t ouptIndex;

};

#endif // MELGENERATETF_H