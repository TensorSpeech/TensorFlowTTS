package com.tensorspeech.tensorflowtts.tts;

import android.media.AudioAttributes;
import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioTrack;
import android.util.Log;

import com.tensorspeech.tensorflowtts.utils.ThreadPoolManager;

import java.util.concurrent.LinkedBlockingQueue;

/**
 * @author {@link "mailto:xuefeng.ding@outlook.com" "Xuefeng Ding"}
 * Created 2020-07-20 18:22
 */
class TtsPlayer {
    private static final String TAG = "TtsPlayer";

    private final AudioTrack mAudioTrack;

    private final static int FORMAT = AudioFormat.ENCODING_PCM_FLOAT;
    private final static int SAMPLERATE = 22050;
    private final static int CHANNEL = AudioFormat.CHANNEL_OUT_MONO;
    private final static int BUFFER_SIZE = AudioTrack.getMinBufferSize(SAMPLERATE, CHANNEL, FORMAT);
    private LinkedBlockingQueue<AudioData> mAudioQueue = new LinkedBlockingQueue<>();
    private AudioData mCurrentAudioData;

    TtsPlayer() {
        mAudioTrack = new AudioTrack(
                new AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_MEDIA)
                        .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC)
                        .build(),
                new AudioFormat.Builder()
                        .setSampleRate(22050)
                        .setEncoding(FORMAT)
                        .setChannelMask(CHANNEL)
                        .build(),
                BUFFER_SIZE,
                AudioTrack.MODE_STREAM, AudioManager.AUDIO_SESSION_ID_GENERATE
        );
        mAudioTrack.play();

        ThreadPoolManager.getInstance().getSingleExecutor("audio").execute(() -> {
            //noinspection InfiniteLoopStatement
            while (true) {
                try {
                    mCurrentAudioData = mAudioQueue.take();
                    Log.d(TAG, "playing: " + mCurrentAudioData.text);
                    int index = 0;
                    while (index < mCurrentAudioData.audio.length && !mCurrentAudioData.isInterrupt) {
                        int buffer = Math.min(BUFFER_SIZE, mCurrentAudioData.audio.length - index);
                        mAudioTrack.write(mCurrentAudioData.audio, index, buffer, AudioTrack.WRITE_BLOCKING);
                        index += BUFFER_SIZE;
                    }
                } catch (Exception e) {
                    Log.e(TAG, "Exception: ", e);
                }
            }
        });
    }

    void play(AudioData audioData) {
        Log.d(TAG, "add audio data to queue: " + audioData.text);
        mAudioQueue.offer(audioData);
    }

    void interrupt() {
        mAudioQueue.clear();
        if (mCurrentAudioData != null) {
            mCurrentAudioData.interrupt();
        }
    }

    static class AudioData {
        private String text;
        private float[] audio;
        private boolean isInterrupt;

        AudioData(String text, float[] audio) {
            this.text = text;
            this.audio = audio;
        }

        private void interrupt() {
            isInterrupt = true;
        }
    }

}
