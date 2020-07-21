/**
 * Copyright [2020] [Xuefeng Ding]
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.tensorspeech.tensorflowtts.module;

import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.File;
import java.nio.FloatBuffer;
import java.util.Arrays;

/**
 * @author {@link "mailto:xuefeng.ding@outlook.com" "Xuefeng Ding"}
 * Created 2020-07-20 17:26
 *
 */
public class MBMelGan extends AbstractModule {
    private static final String TAG = "MBMelGan";
    private Interpreter vocoder;

    public MBMelGan(String modulePath) {
        try {
            vocoder = new Interpreter(new File(modulePath), getOption());
            int input = vocoder.getInputTensorCount();
            for (int i = 0; i < input; i++) {
                Tensor inputTensor = vocoder.getInputTensor(i);
                Log.d(TAG, "input:" + i
                        + " name:" + inputTensor.name()
                        + " shape:" + Arrays.toString(inputTensor.shape()) +
                        " dtype:" + inputTensor.dataType());
            }

            int output = vocoder.getOutputTensorCount();
            for (int i = 0; i < output; i++) {
                Tensor outputTensor = vocoder.getOutputTensor(i);
                Log.d(TAG, "output:" + i
                        + " name:" + outputTensor.name()
                        + " shape:" + Arrays.toString(outputTensor.shape())
                        + " dtype:" + outputTensor.dataType());
            }
            Log.d(TAG, "successfully init");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    public float[] getAudio(TensorBuffer input) {
        vocoder.resizeInput(0, input.getShape());
        vocoder.allocateTensors();

        FloatBuffer outputBuffer = FloatBuffer.allocate(350000);

        long time = System.currentTimeMillis();
        vocoder.run(input.getBuffer(), outputBuffer);
        Log.d(TAG, "time cost: " + (System.currentTimeMillis() - time));

        float[] audioArray = new float[outputBuffer.position()];
        outputBuffer.rewind();
        outputBuffer.get(audioArray);
        return audioArray;
    }
}
