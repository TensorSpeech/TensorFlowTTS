# Fast speech 2 multi-speaker

## Prepare
Everything is done from main repo folder so TensorflowTTS/
0. Optional* Download and prepare libritts (helper to prepare libri in examples/fastspeech2_multispeaker/libri_experiment/prepare_libri.ipynb)
1. Extract Duration (use examples/mfa_extraction or pretrained tacotron2) 
2. Optional* build docker `bash examples/fastspeech2_multispeaker/scripts/build.sh`
3. Optional* run docker `bash examples/fastspeech2_multispeaker/scripts/interactive.sh`
4. Change CharactorDurationF0EnergyMelDataset speaker mapper in fastspeech2_dataset to match your dataset (if you use libri with mfa_extraction you didnt need to change anything)
4. Change train.sh to match your dataset and run `bash examples/fastspeech2_multispeaker/scripts/train.sh`

