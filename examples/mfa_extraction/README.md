# MFA based extraction for FastSpeech 

## Prepare
Everything is done from main repo folder so TensorflowTTS/

0. Optional* Modify MFA scripts to work with your language (https://montreal-forced-aligner.readthedocs.io/en/latest/pretrained_models.html)

1. Download pretrained mfa, lexicon and run extract textgrids:

- ```
  bash examples/mfa_extraction/scripts/prepare_mfa.sh
  ```

- ```
  python examples/mfa_extraction/run_mfa.py \
    --corpus_directory ./libritts \
    --output_directory ./mfa/parsed \
    --jobs 8
  ```

  After this step, the TextGrids is allocated at `./mfa/parsed`.

2. Extract duration from textgrid files:
- ```
  python examples/mfa_extraction/txt_grid_parser.py \
    --yaml_path examples/fastspeech2_libritts/conf/fastspeech2libritts.yaml \
    --dataset_path ./libritts \
    --text_grid_path ./mfa/parsed \
    --output_durations_path ./libritts/durations \
    --sample_rate 24000 
  ```

- Dataset structure after finish this step:
    ```
    |- TensorFlowTTS/
    |   |- LibriTTS/
    |   |-  |- train-clean-100/
    |   |-  |- SPEAKERS.txt
    |   |-  |- ...
    |   |- dataset/
    |   |-  |- 200/
    |   |-  |-  |- 200_124139_000001_000000.txt
    |   |-  |-  |- 200_124139_000001_000000.wav
    |   |-  |-  |- ...
    |   |-  |- 250/
    |   |-  |- ...
    |   |-  |- durations/
    |   |-  |- train.txt
    |   |- tensorflow_tts/
    |       |- models/
    |       |- ...
    ``` 
3. Optional* add your own dataset parser based on tensorflow_tts/processor/experiment/example_dataset.py ( If base processor dataset didnt match yours )

4. Run preprocess and normalization (Step 4,5 in `examples/fastspeech2_libritts/README.MD`)

5. Run fix mismatch to fix few frames difference in audio and duration files:

- ```
  python examples/mfa_extraction/fix_mismatch.py \
    --base_path ./dump \
    --trimmed_dur_path ./dataset/trimmed-durations \
    --dur_path ./dataset/durations
  ```

## Problems with MFA extraction
Looks like MFA have problems with trimmed files it works better (in my experiments) with ~100ms of silence at start and end

Short files can get a lot of false positive like only silence extraction (LibriTTS example) so i would get only samples >2s
