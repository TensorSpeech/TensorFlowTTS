# TensorflowTTS C++ Inference 

This contains code (and libs) necessary to make portable programs for inference with FastSpeech2 (MFA-aligned phonetic) and MB-MelGAN on desktop, along with a simple example.

The program requires two things:
1. An exported and packed TTS model (FS2 + MB-MelGAN). Colab notebook linked below will show
2. A G2P-RNN model. See [here](https://github.com/ZDisket/TensorVox/tree/master/g2p_train) for details.

If you want to convert your model to the format that this program expects, you can check out the notebook: [<img src="https://colab.research.google.com/assets/colab-badge.svg">](https://colab.research.google.com/drive/1EEkJSq9Koo_eI0Sotc_t_CTducybdV9h?usp=sharing)

It includes a set of easily understandable and modular classes including a simple English text preprocessor, so you can easily copy and integrate them into your program.

Inference is even easier than in Python. First you need a Phonemizer, then the voice.

    #include "Voice.h"
    std::string LanguagePath = "g2p/English"
    std::string VoicePath = "LJ";
    
    Phonemizer Phony;
    Phony.Initialize(LanguagePath);
    
    // The Voice class takes a pointer to the Phonemizer to use it. 
    // Don't let it go out of scope!
    Voice LJSpeech(VoicePath,VoicePath,&Phony);
    
    std::vector<float> AudioData = LJSpeech.Vocalize("I love see plus plus" + LJSpeech.GetInfo().EndPadding);
    VoxUtil::ExportWAV("voc1.wav", AudioData, LJSpeech.GetInfo().SampleRate);



# Using the demo

The demo program is available for download to use for Windows and Linux (Ubuntu 18.04), both x64.
It can take command line arguments (see code for details), but defaults should be fine for mere LJSpeech testing.

To use it, do the following depending on platform:

## Using the precompiled demo for Windows
 1. Download the [Windows x64 binary and LJSpeech model](https://drive.google.com/file/d/19ZaiBDtEkyrov_SfVHQUIHgVjVWv2Msu/view?usp=sharing)
 2. Extract to whatever directory you like
 3. Run

## Using the precompiled demo for Linux
Tested in Ubuntu 18.04 LTS
1. Download the [Linux x64 binary and LJSpeech model](https://drive.google.com/file/d/1IgN9KMq2ccF-QSJX_Z1n94mtMDitnFs4/view?usp=sharing)
2. Extract to whatever directory you like
3. Navigate with terminal
4. `LD_LIBRARY_PATH=lib ./TensorflowTTSCppInference`

For compiling it yourself, see **Compiling** below

# Compiling
Compiling the demo depends on what platform. Currently two have been tested:
1. Windows 10 x64; MSVC 2019 
2. Linux(Ubuntu) x64: GCC 7.5.0

Note that to test out your shiny new build afterwards you'll have to download the LJSpeech model (or make one yourself), it's bundled in any of the above precompiled demo download links.

## Dependencies
Download the [dependencies](https://drive.google.com/file/d/167LJXVO2dbFVc1Mmqacrq4LBaUIG9paH/view?usp=sharing) (hint: it's just Tensorflow C API) and drop the deps folder into the same place as the .sln and .pro; it has both Linux and Windows versions.

The rest (such as CppFlow and AudioFile) are included in the source code

## Windows
Use the Visual Studio solution file.

## Ubuntu
Tested with compiler `gcc version 7.5.0 (Ubuntu 7.5.0-3ubuntu1~18.04)` . 
1. `sudo apt install qt5-default`
2. `qmake TensorflowTTSCppInference.pro`
3. `make`


### Notes when compiling
 1. Tensorflow library malfunctions in debug builds, so only build release.

## Externals (and thanks)

 - **Tensorflow C API**: [https://www.tensorflow.org/install/lang_c](https://www.tensorflow.org/install/lang_c)
 - **CppFlow** (TF C API -> C++ wrapper): [https://github.com/serizba/cppflow](https://github.com/serizba/cppflow) 
 - **AudioFile** (for WAV export): [https://github.com/adamstark/AudioFile](https://github.com/adamstark/AudioFile)
 - [nlohmann/json: JSON for Modern C++ ](https://github.com/nlohmann/json)
 - [jarro2783/cxxopts: Lightweight C++ command line option parser)](https://github.com/jarro2783/cxxopts)
 

