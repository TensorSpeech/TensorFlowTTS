package com.tensorspeech.tensorflowtts;

import android.os.Bundle;
import android.util.Log;

import com.tensorspeech.tensorflowtts.module.FastSpeech2;
import com.tensorspeech.tensorflowtts.module.MBMelGan;

import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import androidx.appcompat.app.AppCompatActivity;

/**
 * @author {@link "mailto:xuefeng.ding@outlook.com" "Xuefeng Ding"}
 * Created 2020-07-20 17:25
 *
 */
public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";
    private FastSpeech2 fastSpeech2;
    private MBMelGan mbMelGan;
    private Player player;


    private final static int[] INPUT_EXAMPLE = {55, 42, 40, 42, 51, 57, 11, 55, 42, 56, 42, 38, 55, 40, 45, 11, 38, 57, 11, 45, 38, 55, 59, 38, 55, 41, 11, 45, 38, 56, 11, 56, 45, 52, 60, 51, 11, 50, 42, 41, 46, 57, 38, 57, 46, 51, 44, 43, 52, 55, 11, 38, 56, 11, 49, 46, 57, 57, 49, 42, 11, 38, 56, 11, 42, 46, 44, 45, 57, 11, 60, 42, 42, 48, 56, 6, 11};
    private final static String FASTSPEECH2_MODULE = "fastspeech2_quant.tflite";
    private final static String VOCODER_MODULE = "mbmelgan.tflite";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        init();

        findViewById(R.id.start).setOnClickListener(v ->
                new Thread(() -> {
                    TensorBuffer output = fastSpeech2.getMelSpectrogram(INPUT_EXAMPLE);

                    float[] audioData = mbMelGan.getAudio(output);

                    player.play(audioData);
                }).start()
        );
    }

    private void init() {
        try {
            copyFile(FASTSPEECH2_MODULE);
            copyFile(VOCODER_MODULE);
        } catch (Exception e) {
            Log.e(TAG, "init: failed to copy files", e);
        }

        fastSpeech2 = new FastSpeech2(getFilesDir().getAbsolutePath() + "/" + FASTSPEECH2_MODULE);
        mbMelGan = new MBMelGan(getFilesDir().getAbsolutePath() + "/" + VOCODER_MODULE);
        player = new Player();
    }

    private void copyFile(String strOutFileName) throws IOException {
        Log.d(TAG, "start copy file " + strOutFileName);
        File file = getFilesDir();

        String tmpFile = file.getAbsolutePath() + "/" + strOutFileName;
        File f = new File(tmpFile);
        if (f.exists()) {
            Log.d(TAG, "file exists " + strOutFileName);
            return;
        }

        InputStream myInput = null;
        OutputStream myOutput = null;
        try {
            myOutput = new FileOutputStream(f);
            myInput = this.getAssets().open(strOutFileName);
            byte[] buffer = new byte[1024];
            int length = myInput.read(buffer);
            while (length > 0) {
                myOutput.write(buffer, 0, length);
                length = myInput.read(buffer);
            }
            myOutput.flush();
            Log.d(TAG, "Copy task successful");
        } catch (Exception e) {
            Log.e(TAG, "copyFile: Failed to copy", e);
        } finally {
            myInput.close();
            myOutput.close();
            Log.d(TAG, "end copy file " + strOutFileName);
        }
    }

}
