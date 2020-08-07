# TensorflowTTS C++ Inference 

This contains code (and libs, if using Windows) necessary to make portable programs for inference with FastSpeech2 (MFA-aligned phonetic) and MB-MelGAN on desktop, along with a simple example. The aforementioned models have to be exported in Tensorflow SavedModel format to be used.

It includes a set of easily understandable and modular classes including a simple English text preprocessor, so you can easily copy and integrate them into your program.

Inference is even easier than in Python.

    #include "Voice.h"
    // This is a folder name. See Voice.h for information 
    Voice LJSpeech("LJ");
    std::vector<float> AudioData = LJSpeech.Vocalize("I love see plus plus");
    VoxUtil::ExportWAV("voc1.wav", AudioData, 22050);

# Using the demo

The demo program is available for download to use for Windows x64. To use it, do the following:

## Using the precompiled demo for Windows
 1. Download the [Windows x64 binary and LJSpeech model](https://drive.google.com/file/d/1JHoR3kfFRcxZwghOXsAWXzhYZ3pnZmcc/view?usp=sharing)
 2. Download the [dependencies](https://drive.google.com/file/d/1ufLQvH-Me2NLmzNBkjcyD13WTyHb35aB/view?usp=sharing)
 3. Extract both, and place `tensorflow.dll` in the same folder as the executable you downloaded in Step 1
 4. Run

For compiling it yourself, see **Compiling** below

# Compiling
Due to being too heavy, some dependencies (include + libs) have been excluded from the repo. You can find them [here](https://drive.google.com/file/d/1ufLQvH-Me2NLmzNBkjcyD13WTyHb35aB/view?usp=sharing) for Windows 64-bit release. Simply put the `deps` folder into the same place as the solution (.sln) file.

### Notes when compiling

 1. With MSVC compiler, you'll have to use `/FORCE` as linker command line option, because otherwise the linker throws `LNK2005` due to OpenFST linking; the included project is already set up for that. Not sure about GCC, but you'll probably have to do something similar.
 2. Tensorflow library malfunctions in debug builds, so only build release.
 3. Using MSVC 2017 (v141) compiler.

## Externals (and thanks)

 - **Phonetisaurus** (text to phoneme): [https://github.com/AdolfVonKleist/Phonetisaurus](https://github.com/AdolfVonKleist/Phonetisaurus) *(ported to Windows)*
 - **OpenFST 1.6.2** (for Windows): [https://github.com/kkm000/openfst](https://github.com/kkm000/openfst)
 - **Tensorflow C API**: [https://www.tensorflow.org/install/lang_c](https://www.tensorflow.org/install/lang_c)
 - **CppFlow** (TF C API -> C++ wrapper): [https://github.com/serizba/cppflow](https://github.com/serizba/cppflow) 
 - **AudioFile** (for WAV export): [https://github.com/adamstark/AudioFile](https://github.com/adamstark/AudioFile)





