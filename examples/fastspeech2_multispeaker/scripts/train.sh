CUDA_VISIBLE_DEVICES=0 python examples/fastspeech2_multispeaker/train_fastspeech2.py \
  --train-dir ./dump/train/ \
  --dev-dir ./dump/valid/ \
  --outdir ./examples/fastspeech2_multispeaker/outdir/ \
  --config ./examples/fastspeech2_multispeaker/conf/fastspeech2.v1.yaml \
  --use-norm 1 \
  --f0-stat ./dump/stats_f0.npy \
  --energy-stat ./dump/stats_energy.npy \
  --mixed_precision 1