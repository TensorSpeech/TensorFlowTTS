# Fast speech 2 multi-speaker english lang based

## Prepare
Everything is done from main repo folder so TensorflowTTS/

0. Optional* [Download](http://www.openslr.org/60/) and prepare libritts (helper to prepare libri in examples/fastspeech2_libritts/libri_experiment/prepare_libri.ipynb)
- Dataset structure after finish this step:
    ```
    |- TensorFlowTTS/
    |   |- LibriTTS/
    |   |-  |- train-clean-100/
    |   |-  |- SPEAKERS.txt
    |   |-  |- ...
    |   |- libritts/
    |   |-  |- 200/
    |   |-  |-  |- 200_124139_000001_000000.txt
    |   |-  |-  |- 200_124139_000001_000000.wav
    |   |-  |-  |- ...
    |   |-  |- 250/
    |   |-  |- ...
    |   |- tensorflow_tts/
    |       |- models/
    |       |- ...
    ``` 
1. Extract Duration (use examples/mfa_extraction or pretrained tacotron2) 
2. Optional* build docker 
- ```
  bash examples/fastspeech2_libritts/scripts/build.sh
  ```
3. Optional* run docker
- ```
  bash examples/fastspeech2_libritts/scripts/interactive.sh
  ```
4. Preprocessing:
- ```
  tensorflow-tts-preprocess --rootdir ./libritts \
    --outdir ./dump_libritts \
    --config preprocess/libritts_preprocess.yaml \
    --dataset libritts
  ```

5. Normalization:
- ```
  tensorflow-tts-normalize --rootdir ./dump_libritts \
    --outdir ./dump_libritts \
    --config preprocess/libritts_preprocess.yaml \
    --dataset libritts
  ```

6. Change CharactorDurationF0EnergyMelDataset speaker mapper in fastspeech2_dataset to match your dataset (if you use libri with mfa_extraction you didnt need to change anything)
7. Change train_libri.sh to match your dataset and run:
- ```
  bash examples/fastspeech2_libritts/scripts/train_libri.sh
  ```
8. Optional* If u have problems with tensor sizes mismatch check step 5 in `examples/mfa_extraction` directory

## Comments

This version is using popular train.txt '|' split used in other repos. Training files should looks like this =>

Wav Path | Text | Speaker Name

Wav Path2 | Text | Speaker Name

