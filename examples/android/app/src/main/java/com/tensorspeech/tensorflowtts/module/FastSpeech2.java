package com.tensorspeech.tensorflowtts.module;

import android.annotation.SuppressLint;
import android.util.Log;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.File;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * @author {@link "mailto:xuefeng.ding@outlook.com" "Xuefeng Ding"}
 * Created 2020-07-20 17:26
 *
 */
public class FastSpeech2 extends AbstractModule {
    private static final String TAG = "FastSpeech2";
    private Interpreter encoder = null;

    public FastSpeech2(String modulePath) {
        try {
            encoder = new Interpreter(new File(modulePath), getOption());
            int input = encoder.getInputTensorCount();
            for (int i = 0; i < input; i++) {
                Tensor inputTensor = encoder.getInputTensor(i);
                Log.d(TAG, "input:" + i +
                        " name:" + inputTensor.name() +
                        " shape:" + Arrays.toString(inputTensor.shape()) +
                        " dtype:" + inputTensor.dataType());
            }

            int output = encoder.getOutputTensorCount();
            for (int i = 0; i < output; i++) {
                Tensor outputTensor = encoder.getOutputTensor(i);
                Log.d(TAG, "output:" + i +
                        " name:" + outputTensor.name() +
                        " shape:" + Arrays.toString(outputTensor.shape()) +
                        " dtype:" + outputTensor.dataType());
            }
            Log.d(TAG, "successfully init");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public TensorBuffer getMelSpectrogram(int[] inputIds) {
        Log.d(TAG, "input id length: " + inputIds.length);
        encoder.resizeInput(0, new int[]{1, inputIds.length});
        encoder.allocateTensors();

        @SuppressLint("UseSparseArrays")
        Map<Integer, Object> outputMap = new HashMap<>();

        FloatBuffer outputBuffer = FloatBuffer.allocate(350000);
        outputMap.put(0, outputBuffer);

        int[][] inputs = new int[1][inputIds.length];
        inputs[0] = inputIds;

        long time = System.currentTimeMillis();
        encoder.runForMultipleInputsOutputs(
                new Object[]{inputs, new int[1][1], new int[]{0}, new float[]{1F}, new float[]{1F}, new float[]{1F}},
                outputMap);
        Log.d(TAG, "time cost: " + (System.currentTimeMillis() - time));

        int size = encoder.getOutputTensor(0).shape()[2];
        int[] shape = {1, outputBuffer.position() / size, size};
        TensorBuffer spectrogram = TensorBuffer.createFixedSize(shape, DataType.FLOAT32);
        float[] outputArray = new float[outputBuffer.position()];
        outputBuffer.rewind();
        outputBuffer.get(outputArray);
        spectrogram.loadArray(outputArray);

        return spectrogram;
    }
}
