package com.tensorspeech.tensorflowtts;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.tensorspeech.tensorflowtts.module.FastSpeech2;
import com.tensorspeech.tensorflowtts.module.MBMelGan;

import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

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
    private final static String FASTSPEECH2_MODULE = "/sdcard/fastspeech2_quant.tflite";
    private final static String VOCODER_MODULE = "/sdcard/mbmelgan.tflite";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED) {
            Log.d(TAG, "onCreate: permission granted");
            init();
        } else {
            Log.e(TAG, "onCreate: permission missing");
            ActivityCompat.requestPermissions(MainActivity.this,
                    new String[]{Manifest.permission.READ_EXTERNAL_STORAGE, Manifest.permission.RECORD_AUDIO},
                    1);
        }

        findViewById(R.id.start).setOnClickListener(v ->
                new Thread(() -> {
                    TensorBuffer output = fastSpeech2.getMelSpectrogram(INPUT_EXAMPLE);

                    float[] audioData = mbMelGan.getAudio(output);

                    player.play(audioData);
                }).start()
        );
    }

    private void init() {
        fastSpeech2 = new FastSpeech2(FASTSPEECH2_MODULE);
        mbMelGan = new MBMelGan(VOCODER_MODULE);
        player = new Player();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        init();
    }
}
