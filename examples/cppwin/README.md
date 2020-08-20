# TensorflowTTS C++ Inference 

This contains code (and libs, if using Windows) necessary to make portable programs for inference with FastSpeech2 (MFA-aligned phonetic) and MB-MelGAN on desktop, along with a simple example. The aforementioned models have to be exported in Tensorflow SavedModel format to be used.

If you want to convert your model to the format that this program expects, you can check out the notebook: [<img src="https://colab.research.google.com/assets/colab-badge.svg">](https://colab.research.google.com/drive/1qRzNLZaZu_TtUYv-OH7N7xXD3vXtxD_z?usp=sharing)

It includes a set of easily understandable and modular classes including a simple English text preprocessor, so you can easily copy and integrate them into your program.

Inference is even easier than in Python.

    #include "Voice.h"
    // This is a folder name. See Voice.h for information 
    Voice LJSpeech("LJ");
    std::vector<float> AudioData = LJSpeech.Vocalize("I love see plus plus");
    VoxUtil::ExportWAV("voc1.wav", AudioData, 22050);

# Using the demo

The demo program is available for download to use for Windows x64 and Linux (Ubuntu 18.04) x64.

To use it, do the following depending on platform:

## Using the precompiled demo for Windows
 1. Download the [Windows x64 binary and LJSpeech model](https://drive.google.com/file/d/1JHoR3kfFRcxZwghOXsAWXzhYZ3pnZmcc/view?usp=sharing)
 2. Download the [dependencies](https://drive.google.com/file/d/1ufLQvH-Me2NLmzNBkjcyD13WTyHb35aB/view?usp=sharing)
 3. Extract both, and place `tensorflow.dll` in the same folder as the executable you downloaded in Step 1
 4. Run

## Using the precompiled demo for Linux
Tested in Kubuntu 18.04 LTS
1. Download the [Linux x64 binary and LJSpeech model](https://drive.google.com/file/d/17wAVsxRpaPogPpzOiEJB8QJjF_wI60q2/view?usp=sharing)
2. Extract to whatever directory you like
3. Navigate with terminal
4. `LD_LIBRARY_PATH=lib ./TensorflowTTSCppInference`


For compiling it yourself, see **Compiling** below

# Compiling
Compiling the demo depends on what platform. Currently two have been tested:
1. Windows 10 x64; MSVC 2017 (v141)
2. Linux(Ubuntu) x64: GCC 7.5.0

Note that to test out your shiny new build afterwards you'll have to download the LJSpeech model (or make one yourself), it's bundled in any of the above precompiled demo download links.

## Windows
Due to being too heavy, some dependencies (include + libs) have been excluded from the repo. You can find them [here](https://drive.google.com/file/d/1ufLQvH-Me2NLmzNBkjcyD13WTyHb35aB/view?usp=sharing) for Windows 64-bit release. Simply put the `deps` folder into the same place as the solution (.sln) file.

## Ubuntu
Tested with compiler `gcc version 7.5.0 (Ubuntu 7.5.0-3ubuntu1~18.04)` . If you've got the same one then you should try my precompiled libs (also includes Tensorflow official C API).
1. Download [Linux dependencies](https://drive.google.com/file/d/1MF4QXq69l8h6nh4h_nKatShobyfbAkiD/view?usp=sharing)  and extract the deps folder in the same place as the .pro
2. `sudo apt install qt5-default`
3. `qmake TensorflowTTSCppInference.pro`
4. `make`


## For other platforms
For Windows x64 and Ubuntu 18.04 x64 (gcc version 7.5.0), precompiled libraries are included. For other platforms, you'll have to download and compile from source the following dependencies:

 1. [libPhonetisaurus](https://github.com/ZDisket/Phonetisaurus)
 2. [OpenFST](http://www.openfst.org/twiki/bin/view/FST/WebHome) (preferably 1.6.2)
 
 For Tensorflow, you need the C API, which you can grab compiled binaries (or view compile instructions) [from here](https://www.tensorflow.org/install/lang_c).
 
 The rest (such as CppFlow and AudioFile) are included in the source code.

### Notes when compiling

 1. With MSVC compiler, you'll have to use `/FORCE` as linker command line option, because otherwise the linker throws `LNK2005` due to OpenFST linking; the included project is already set up for that. Not sure about GCC, but you'll probably have to do something similar.
 2. Tensorflow library malfunctions in debug builds, so only build release.
 3. Using MSVC 2017 (v141) compiler in Windows.

## Externals (and thanks)

 - **Phonetisaurus** (text to phoneme): [https://github.com/AdolfVonKleist/Phonetisaurus](https://github.com/AdolfVonKleist/Phonetisaurus) *(ported to Windows)*
 - **OpenFST 1.6.2** (for Windows): [https://github.com/kkm000/openfst](https://github.com/kkm000/openfst)
 - **Tensorflow C API**: [https://www.tensorflow.org/install/lang_c](https://www.tensorflow.org/install/lang_c)
 - **CppFlow** (TF C API -> C++ wrapper): [https://github.com/serizba/cppflow](https://github.com/serizba/cppflow) 
 - **AudioFile** (for WAV export): [https://github.com/adamstark/AudioFile](https://github.com/adamstark/AudioFile)

