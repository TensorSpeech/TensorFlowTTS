# iOS Demo

This app demonstrates using FastSpeech2 and MB MelGAN models on iOS.

## How to build

Download LJ Speech TFLite models from https://github.com/luan78zaoha/TTS_tflite_cpp/releases/tag/0.1.0 and unpack into TF_TTS_Demo directory containing Swift files.

It uses [CocoaPods](https://cocoapods.org) to link with TensorFlowSwift.

```
pod install
open TF_TTS_Demo.xcworkspace
```

