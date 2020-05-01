#!/bin/bash

# This code is modified from https://github.com/kan-bayashi/ParallelWaveGAN.

. ./cmd.sh || exit 1;
. ./path.sh || exit 1;

# basic settings
stage=-1       # stage to start
stop_stage=100 # stage to stop
verbose=1      # verbosity level (lower is less info)
n_gpus=1       # number of gpus in training
n_jobs=16      # number of parallel jobs in feature extraction

# NOTE(kan-bayashi): renamed to conf to avoid conflict in parse_options.sh
conf=conf/melgan.v1.yaml

# directory path setting
download_dir=downloads # direcotry to save downloaded files
dumpdir=dump           # directory to dump features

# training related setting
tag=""     # tag for directory to save model
resume=""  # checkpoint path to resume training
           # (e.g. <path>/<to>/checkpoint-10000steps.pkl)
generator_mixed_precision=0  # use mixed precision for generator or not.
discriminator_mixed_precision=0  # use mixed precision for discriminator or not.

# decoding related setting
checkpoint="" # checkpoint path to be used for decoding
              # if not provided, the latest one will be used
              # (e.g. <path>/<to>/checkpoint-400000steps.pkl)


# shellcheck disable=SC1091
. parse_options.sh || exit 1;

train_set="train_nodev" # name of training data directory
dev_set="dev"           # name of development data direcotry
eval_set="eval"         # name of evaluation data direcotry

set -euo pipefail

if [ "${stage}" -le -1 ] && [ "${stop_stage}" -ge -1 ]; then
    echo -e "\e[92mStage -1: Data download\e[39m"
    local/data_download.sh "${download_dir}"
fi

if [ "${stage}" -le 0 ] && [ "${stop_stage}" -ge 0 ]; then
    echo -e "\e[92mStage 0: Data preparation\e[39m"
    local/data_prep.sh \
        --train_set "${train_set}" \
        --dev_set "${dev_set}" \
        --eval_set "${eval_set}" \
        "${download_dir}/LJSpeech-1.1" data
fi

stats_ext=$(grep -q "hdf5" <(yq ".format" "${conf}") && echo "h5" || echo "npy")
if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    echo -e "\e[92mStage 1: Feature extraction\e[39m"
    SECONDS=0
    # extract raw features
    pids=()
    for name in "${train_set}" "${dev_set}" "${eval_set}"; do
    (
        [ ! -e "${dumpdir}/${name}/raw" ] && mkdir -p "${dumpdir}/${name}/raw"
        echo -e "Feature extraction start. See the progress via ${dumpdir}/${name}/raw/preprocessing.*.log."
        make_subset_data.sh "data/${name}" "${n_jobs}" "${dumpdir}/${name}/raw"
        ${train_cmd} JOB=1:${n_jobs} "${dumpdir}/${name}/raw/preprocessing.JOB.log" \
            tensorflow-tts-preprocess \
                --config "${conf}" \
                --scp "${dumpdir}/${name}/raw/wav.JOB.scp" \
                --dumpdir "${dumpdir}/${name}/raw/dump.JOB" \
                --verbose "${verbose}"
        echo -e "Successfully finished feature extraction of ${name} set."
    ) &
    pids+=($!)
    done
    i=0; for pid in "${pids[@]}"; do wait "${pid}" || ((++i)); done
    [ "${i}" -gt 0 ] && echo "$0: ${i} background jobs are failed." && exit 1;
    echo -e "Successfully finished feature extraction."

    # calculate statistics for normalization
    echo -e "\e[92mStatistics computation start. See the progress via ${dumpdir}/${train_set}/compute_statistics.log.\e[39m"
    ${train_cmd} "${dumpdir}/${train_set}/compute_statistics.log" \
        tensorflow-tts-compute-statistics \
            --config "${conf}" \
            --rootdir "${dumpdir}/${train_set}/raw" \
            --dumpdir "${dumpdir}/${train_set}" \
            --verbose "${verbose}"
    echo -e "Successfully finished calculation of statistics."

    # normalize and dump them
    pids=()
    for name in "${train_set}" "${dev_set}" "${eval_set}"; do
    (
        [ ! -e "${dumpdir}/${name}/norm" ] && mkdir -p "${dumpdir}/${name}/norm"
        echo -e "\e[92mNomalization start. See the progress via ${dumpdir}/${name}/norm/normalize.*.log.\e[39m"
        ${train_cmd} JOB=1:${n_jobs} "${dumpdir}/${name}/norm/normalize.JOB.log" \
            tensorflow-tts-normalize \
                --config "${conf}" \
                --stats "${dumpdir}/${train_set}/stats.${stats_ext}" \
                --rootdir "${dumpdir}/${name}/raw/dump.JOB" \
                --dumpdir "${dumpdir}/${name}/norm/dump.JOB" \
                --verbose "${verbose}"
        echo -e "Successfully finished normalization of ${name} set."
    ) &
    pids+=($!)
    done
    i=0; for pid in "${pids[@]}"; do wait "${pid}" || ((++i)); done
    [ "${i}" -gt 0 ] && echo "$0: ${i} background jobs are failed." && exit 1;
    echo -e "Successfully finished normalization."

    duration=$SECONDS
    echo -e "\e[91mFeature extraction stage took \e[39m$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed.\e[39m"
fi

if [ -z "${tag}" ]; then
    expdir="exp/${train_set}_ljspeech_$(basename "${conf}" .yaml)"
else
    expdir="exp/${train_set}_ljspeech_${tag}"
fi
if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    echo "Stage 2: Network training"
    [ ! -e "${expdir}" ] && mkdir -p "${expdir}"
    cp "${dumpdir}/${train_set}/stats.${stats_ext}" "${expdir}"
    train="tensorflow-tts-train-melgan"
    echo "Training start. See the progress via ${expdir}/train.log."
    ${cuda_cmd} --gpu "${n_gpus}" "${expdir}/train.log" \
        ${train} \
            --config "${conf}" \
            --train-dumpdir "${dumpdir}/${train_set}/norm" \
            --dev-dumpdir "${dumpdir}/${dev_set}/norm" \
            --outdir "${expdir}" \
            --resume "${resume}" \
            --verbose "${verbose}" \
            --generator_mixed_precision "${generator_mixed_precision}" \
            --discriminator_mixed_precision "${discriminator_mixed_precision}"
    echo "Successfully finished training."
fi