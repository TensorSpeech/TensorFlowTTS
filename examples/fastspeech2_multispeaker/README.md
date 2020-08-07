# Fast speech 2 multi-speaker

## Prepare
Everything is done from main repo folder so TensorflowTTS/
0. Optional* Download and prepare libritts (helper to prepare libri in examples/fastspeech2_multispeaker/libri_experiment/prepare_libri.ipynb)
1. Extract Duration (use examples/mfa_extraction or pretrained tacotron2) 
2. Optional* build docker `bash examples/fastspeech2_multispeaker/scripts/build.sh`
3. Optional* run docker `bash examples/fastspeech2_multispeaker/scripts/interactive.sh`
4. Run `tensorflow-tts-preprocess --rootdir ./dataset --outdir ./dump --config preprocess/preprocess_multispeaker.yaml --dataset multispeaker`
5. Run `tensorflow-tts-normalize --rootdir ./dump --outdir ./dump --config preprocess/preprocess_multispeaker.yaml --dataset multispeaker`
6. Change CharactorDurationF0EnergyMelDataset speaker mapper in fastspeech2_dataset to match your dataset (if you use libri with mfa_extraction you didnt need to change anything)
7. Change train.sh to match your dataset and run `bash examples/fastspeech2_multispeaker/scripts/train.sh`
8. Optional* If u have problems with tensor sizes mismatch check examples/mfa_extraction directory

## Comments

This version is using popular train.txt '|' split used in other repos. Training files should looks like this =>

Wav Path | Text | Speaker Name

Wav Path2 | Text | Speaker Name

