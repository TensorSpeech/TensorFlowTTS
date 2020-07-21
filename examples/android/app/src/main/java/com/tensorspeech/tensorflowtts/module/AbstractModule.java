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
