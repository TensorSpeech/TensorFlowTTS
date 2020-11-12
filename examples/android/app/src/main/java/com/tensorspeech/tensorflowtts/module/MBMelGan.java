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
    private Interpreter mModule;

    public MBMelGan(String modulePath) {
        try {
            mModule = new Interpreter(new File(modulePath), getOption());
            int input = mModule.getInputTensorCount();
            for (int i = 0; i < input; i++) {
                Tensor inputTensor = mModule.getInputTensor(i);
                Log.d(TAG, "input:" + i
                        + " name:" + inputTensor.name()
                        + " shape:" + Arrays.toString(inputTensor.shape()) +
                        " dtype:" + inputTensor.dataType());
            }

            int output = mModule.getOutputTensorCount();
            for (int i = 0; i < output; i++) {
                Tensor outputTensor = mModule.getOutputTensor(i);
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
        mModule.resizeInput(0, input.getShape());
        mModule.allocateTensors();

        FloatBuffer outputBuffer = FloatBuffer.allocate(350000);

        long time = System.currentTimeMillis();
        mModule.run(input.getBuffer(), outputBuffer);
        Log.d(TAG, "time cost: " + (System.currentTimeMillis() - time));

        float[] audioArray = new float[outputBuffer.position()];
        outputBuffer.rewind();
        outputBuffer.get(audioArray);
        return audioArray;
    }
}
