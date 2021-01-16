#include "TfliteBase.h"

TfliteBase::TfliteBase(const char* modelFilename)
{
    interpreterBuild(modelFilename);
}

TfliteBase::~TfliteBase()
{
    ;
}

void TfliteBase::interpreterBuild(const char* modelFilename)
{
    model = tflite::FlatBufferModel::BuildFromFile(modelFilename);

    TFLITE_MINIMAL_CHECK(model != nullptr);

    tflite::InterpreterBuilder builder(*model, resolver);

    builder(&interpreter);

    TFLITE_MINIMAL_CHECK(interpreter != nullptr);
}
