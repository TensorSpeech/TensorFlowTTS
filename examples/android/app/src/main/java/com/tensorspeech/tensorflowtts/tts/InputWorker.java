package com.tensorspeech.tensorflowtts.tts;

import android.util.Log;

import com.tensorspeech.tensorflowtts.dispatcher.TtsStateDispatcher;
import com.tensorspeech.tensorflowtts.module.FastSpeech2;
import com.tensorspeech.tensorflowtts.module.MBMelGan;
import com.tensorspeech.tensorflowtts.utils.Processor;
import com.tensorspeech.tensorflowtts.utils.ThreadPoolManager;

import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.util.Arrays;
import java.util.concurrent.LinkedBlockingQueue;

/**
 * @author {@link "mailto:xuefeng.ding@outlook.com" "Xuefeng Ding"}
 * Created 2020-07-28 14:25
 */
class InputWorker {
    private static final String TAG = "InputWorker";

    private LinkedBlockingQueue<InputText> mInputQueue = new LinkedBlockingQueue<>();
    private InputText mCurrentInputText;
    private FastSpeech2 mFastSpeech2;
    private MBMelGan mMBMelGan;
    private Processor mProcessor;
    private TtsPlayer mTtsPlayer;

    InputWorker(String fastspeech, String vocoder) {
        mFastSpeech2 = new FastSpeech2(fastspeech);
        mMBMelGan = new MBMelGan(vocoder);
        mProcessor = new Processor();
        mTtsPlayer = new TtsPlayer();

        ThreadPoolManager.getInstance().getSingleExecutor("worker").execute(() -> {
            //noinspection InfiniteLoopStatement
            while (true) {
                try {
                    mCurrentInputText = mInputQueue.take();
                    Log.d(TAG, "processing: " + mCurrentInputText.INPUT_TEXT);
                    TtsStateDispatcher.getInstance().onTtsStart(mCurrentInputText.INPUT_TEXT);
                    mCurrentInputText.proceed();
                    TtsStateDispatcher.getInstance().onTtsStop();
                } catch (Exception e) {
                    Log.e(TAG, "Exception: ", e);
                }
            }
        });
    }

    void processInput(String inputText, float speed) {
        Log.d(TAG, "add to queue: " + inputText);
        mInputQueue.offer(new InputText(inputText, speed));
    }

    void interrupt() {
        mInputQueue.clear();
        if (mCurrentInputText != null) {
            mCurrentInputText.interrupt();
        }
        mTtsPlayer.interrupt();
    }


    private class InputText {
        private final String INPUT_TEXT;
        private final float SPEED;
        private boolean isInterrupt;

        private InputText(String inputText, float speed) {
            this.INPUT_TEXT = inputText;
            this.SPEED = speed;
        }

        private void proceed() {
            String[] sentences = INPUT_TEXT.split("[.,]");
            Log.d(TAG, "speak: " + Arrays.toString(sentences));

            for (String sentence : sentences) {

                long time = System.currentTimeMillis();

                int[] inputIds = mProcessor.textToIds(sentence);

                TensorBuffer output = mFastSpeech2.getMelSpectrogram(inputIds, SPEED);

                if (isInterrupt) {
                    Log.d(TAG, "proceed: interrupt");
                    return;
                }

                long encoderTime = System.currentTimeMillis();

                float[] audioData = mMBMelGan.getAudio(output);

                if (isInterrupt) {
                    Log.d(TAG, "proceed: interrupt");
                    return;
                }

                long vocoderTime = System.currentTimeMillis();

                Log.d(TAG, "Time cost: " + (encoderTime - time) + "+" + (vocoderTime - encoderTime) + "=" + (vocoderTime - time));

                mTtsPlayer.play(new TtsPlayer.AudioData(sentence, audioData));
            }
        }

        private void interrupt() {
            this.isInterrupt = true;
        }
    }
}