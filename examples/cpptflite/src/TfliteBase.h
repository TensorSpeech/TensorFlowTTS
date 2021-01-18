#ifndef TFLITEBASE_H
#define TFLITEBASE_H

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#define TFLITE_MINIMAL_CHECK(x)                              \
    if (!(x)) {                                              \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
    }

typedef struct 
{
    float *melData;
    std::vector<int32_t> melShape;
    int32_t bytes;
} MelGenData;

class TfliteBase
{
public:
    uint32_t int_size   = sizeof(int32_t);
    uint32_t float_size = sizeof(float);

    std::unique_ptr<tflite::Interpreter> interpreter;

    TfliteBase(const char* modelFilename);
    ~TfliteBase();

private:
    std::unique_ptr<tflite::FlatBufferModel> model;
    tflite::ops::builtin::BuiltinOpResolver resolver;

    void interpreterBuild(const char* modelFilename);
};

#endif // TFLITEBASE_H