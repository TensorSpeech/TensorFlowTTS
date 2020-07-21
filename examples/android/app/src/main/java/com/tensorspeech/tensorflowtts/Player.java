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
