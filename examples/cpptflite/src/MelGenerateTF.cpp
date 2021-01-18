#include <iostream>
#include "MelGenerateTF.h"

MelGenData MelGenerateTF::infer(const std::vector<int32_t> inputIds)
{

    MelGenData output;

    int32_t idsLen = inputIds.size();

    std::vector<std::vector<int32_t>> inputIndexsShape{ {1, idsLen}, {1}, {1}, {1}, {1} };

    int32_t shapeI = 0;
    for (auto index : inputIndexs)
    {
        interpreter->ResizeInputTensor(index, inputIndexsShape[shapeI]);
        shapeI++;
    }

    TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);

    int32_t* input_ids_ptr = interpreter->typed_tensor<int32_t>(inputIndexs[0]);
    memcpy(input_ids_ptr, inputIds.data(), int_size * idsLen);

    int32_t* speaker_ids_ptr = interpreter->typed_tensor<int32_t>(inputIndexs[1]);
    memcpy(speaker_ids_ptr, _speakerId.data(), int_size);

    float* speed_ratios_ptr = interpreter->typed_tensor<float>(inputIndexs[2]);
    memcpy(speed_ratios_ptr, _speedRatio.data(), float_size);

    float* speed_ratios2_ptr = interpreter->typed_tensor<float>(inputIndexs[3]);
    memcpy(speed_ratios2_ptr, _f0Ratio.data(), float_size);

    float* speed_ratios3_ptr = interpreter->typed_tensor<float>(inputIndexs[4]);
    memcpy(speed_ratios3_ptr, _enegyRatio.data(), float_size);

    TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);

    TfLiteTensor* melGenTensor = interpreter->tensor(ouptIndex);

    for (int i=0; i<melGenTensor->dims->size; i++)
    {
        output.melShape.push_back(melGenTensor->dims->data[i]);
    }

    output.bytes = melGenTensor->bytes;

    output.melData = interpreter->typed_tensor<float>(ouptIndex);

    return output;
}