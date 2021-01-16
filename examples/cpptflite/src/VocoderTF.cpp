#include "VocoderTF.h"

std::vector<float> VocoderTF::infer(const MelGenData mel)
{
    std::vector<float> audio;

    interpreter->ResizeInputTensor(inputIndex, mel.melShape);
    TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);

    float* melDataPtr = interpreter->typed_input_tensor<float>(inputIndex);
    memcpy(melDataPtr, mel.melData, mel.bytes);

    TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);

    TfLiteTensor* audioTensor = interpreter->tensor(outputIndex);

    float* outputPtr = interpreter->typed_output_tensor<float>(0);

    int32_t audio_len = audioTensor->bytes / float_size;

    for (int i=0; i<audio_len; ++i)
    {
        audio.push_back(outputPtr[i]);
    }

    return audio;
}
