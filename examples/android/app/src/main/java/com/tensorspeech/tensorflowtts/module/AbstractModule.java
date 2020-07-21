package com.tensorspeech.tensorflowtts.module;

import org.tensorflow.lite.Interpreter;

/**
 * @author {@link "mailto:xuefeng.ding@outlook.com" "Xuefeng Ding"}
 * Created 2020-07-20 17:25
 *
 */
public abstract class AbstractModule {

    public Interpreter.Options getOption() {
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(5);
        return options;
    }
}
