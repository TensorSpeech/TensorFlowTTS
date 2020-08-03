package com.tensorspeech.tensorflowtts.tts;

import android.content.Context;
import android.util.Log;

import com.tensorspeech.tensorflowtts.dispatcher.TtsStateDispatcher;
import com.tensorspeech.tensorflowtts.utils.ThreadPoolManager;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;

/**
 * @author {@link "mailto:xuefeng.ding@outlook.com" "Xuefeng Ding"}
 * Created 2020-07-28 14:25
 */
public class TtsManager {
    private static final String TAG = "TtsManager";

    private static final Object INSTANCE_WRITE_LOCK = new Object();

    private static volatile TtsManager instance;

    public static TtsManager getInstance() {
        if (instance == null) {
            synchronized (INSTANCE_WRITE_LOCK) {
                if (instance == null) {
                    instance = new TtsManager();
                }
            }
        }
        return instance;
    }

    private InputWorker mWorker;

    private final static String FASTSPEECH2_MODULE = "fastspeech2_quant.tflite";
    private final static String MELGAN_MODULE = "mbmelgan.tflite";

    public void init(Context context) {
        ThreadPoolManager.getInstance().getSingleExecutor("init").execute(() -> {
            try {
                String fastspeech = copyFile(context, FASTSPEECH2_MODULE);
                String vocoder = copyFile(context, MELGAN_MODULE);
                mWorker = new InputWorker(fastspeech, vocoder);
            } catch (Exception e) {
                Log.e(TAG, "mWorker init failed", e);
            }

            TtsStateDispatcher.getInstance().onTtsReady();
        });
    }

    private String copyFile(Context context, String strOutFileName) {
        Log.d(TAG, "start copy file " + strOutFileName);
        File file = context.getFilesDir();

        String tmpFile = file.getAbsolutePath() + "/" + strOutFileName;
        File f = new File(tmpFile);
        if (f.exists()) {
            Log.d(TAG, "file exists " + strOutFileName);
            return f.getAbsolutePath();
        }

        try (OutputStream myOutput = new FileOutputStream(f);
             InputStream myInput = context.getAssets().open(strOutFileName)) {
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
            Log.d(TAG, "end copy file " + strOutFileName);
        }
        return f.getAbsolutePath();
    }

    public void stopTts() {
        mWorker.interrupt();
    }

    public void speak(String inputText, float speed, boolean interrupt) {
        if (interrupt) {
            stopTts();
        }

        ThreadPoolManager.getInstance().execute(() ->
                mWorker.processInput(inputText, speed));
    }

}
