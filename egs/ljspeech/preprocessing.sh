tensorflow-tts-preprocess --rootdir ./datasets/ --outdir ./dump/ --conf conf/preprocess.yaml
tensorflow-tts-compute-statistics --rootdir ./dump/train/ --outdir ./dump --config conf/preprocess.yaml
tensorflow-tts-normalize --rootdir ./dump --outdir ./dump --stats ./dump/stats.npy --config conf/preprocess.yaml

