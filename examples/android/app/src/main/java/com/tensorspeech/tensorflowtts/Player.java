package com.tensorspeech.tensorflowtts;

import android.media.AudioAttributes;
import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioTrack;

/**
 * @author {@link "mailto:xuefeng.ding@outlook.com" "Xuefeng Ding"}
 * Created 2020-07-20 18:22
 *
 */
public class Player {
    private final AudioTrack audioTrack;

    private final static int FORMAT = AudioFormat.ENCODING_PCM_FLOAT;
    private final static int SAMPLERATE = 22050;
    private final static int CHANNEL = AudioFormat.CHANNEL_OUT_MONO;
    private final static int BUFFER_SIZE = AudioTrack.getMinBufferSize(SAMPLERATE, CHANNEL, FORMAT);

    public Player() {
        audioTrack = new AudioTrack(
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
        audioTrack.play();
    }

    public void play(float[] audioData) {
        int index = 0;
        while (index < audioData.length) {
            int buffer = Math.min(BUFFER_SIZE, audioData.length - index);
            audioTrack.write(audioData, index, buffer, AudioTrack.WRITE_BLOCKING);
            index += BUFFER_SIZE;
        }
    }
}
