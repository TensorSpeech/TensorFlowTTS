package com.tensorspeech.tensorflowtts.dispatcher;

import android.os.Handler;
import android.os.Looper;
import android.util.Log;

import java.util.concurrent.CopyOnWriteArrayList;

/**
 * @author {@link "mailto:xuefeng.ding@outlook.com" "Xuefeng Ding"}
 * Created 2020-07-28 14:25
 */
public class TtsStateDispatcher {
    private static final String TAG = "TtsStateDispatcher";
    private static volatile TtsStateDispatcher instance;
    private static final Object INSTANCE_WRITE_LOCK = new Object();

    public static TtsStateDispatcher getInstance() {
        if (instance == null) {
            synchronized (INSTANCE_WRITE_LOCK) {
                if (instance == null) {
                    instance = new TtsStateDispatcher();
                }
            }
        }
        return instance;
    }

    private final Handler handler = new Handler(Looper.getMainLooper());

    private CopyOnWriteArrayList<OnTtsStateListener> mListeners = new CopyOnWriteArrayList<>();

    public void release() {
        Log.d(TAG, "release: ");
        mListeners.clear();
    }

    public void addListener(OnTtsStateListener listener) {
        if (mListeners.contains(listener)) {
            return;
        }
        Log.d(TAG, "addListener: " + listener.getClass());
        mListeners.add(listener);
    }

    public void removeListener(OnTtsStateListener listener) {
        if (mListeners.contains(listener)) {
            Log.d(TAG, "removeListener: " + listener.getClass());
            mListeners.remove(listener);
        }
    }

    public void onTtsStart(String text){
        Log.d(TAG, "onTtsStart: ");
        if (!mListeners.isEmpty()) {
            for (OnTtsStateListener listener : mListeners) {
                handler.post(() -> listener.onTtsStart(text));
            }
        }
    }

    public void onTtsStop(){
        Log.d(TAG, "onTtsStop: ");
        if (!mListeners.isEmpty()) {
            for (OnTtsStateListener listener : mListeners) {
                handler.post(listener::onTtsStop);
            }
        }
    }

    public void onTtsReady(){
        Log.d(TAG, "onTtsReady: ");
        if (!mListeners.isEmpty()) {
            for (OnTtsStateListener listener : mListeners) {
                handler.post(listener::onTtsReady);
            }
        }
    }
}
