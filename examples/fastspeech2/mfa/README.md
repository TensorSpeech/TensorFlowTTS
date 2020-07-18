# Using alignments from Montreal Forced Aligner to train
This guide will be assuming you're using LJSpeech or an English dataset formatted exactly like it. Note that for optimal performance your audio has to be well trimmed.
**Read this guide before doing preprocessing**

## Setting up MFA
You'll need to download Montreal Forced Aligner; the following commands will work for 64-bit Linux.

We'll be using the LibriSpeech lexicon as a dictionary, this should cover all words on LJSpeech.
```
wget https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/releases/download/v1.1.0-beta.2/montreal-forced-aligner_linux.tar.gz
tar -zxvf montreal-forced-aligner_linux.tar.gz
wget http://www.openslr.org/resources/11/librispeech-lexicon.txt -O montreal-forced-aligner/pretrained_models/librispeech-lexicon.txt
```
**Note that if your dataset contains words not present in the LibriSpeech lexicon stuff will fail**. 

## Using MFA

First, we'll need to set up the files that MFA requires to align. Run this script

```
python examples/fastspeech2/mfa/premfa.py
```
Note that the above script does take some arguments but the defaults should be fine for LJSpeech (or your dataset formatted exactly like it)

Now, run the aligner. It takes the path where the wavs and transcriptions are, the dictionary, pretrained model, and the output directory.

```
./montreal-forced-aligner/bin/mfa_align ./datasets/wavs ./montreal-forced-aligner/pretrained_models/librispeech-lexicon.txt ./montreal-forced-aligner/pretrained_models/english.zip ./TextGrids -j 8
```

## After MFA

After aligning, run the post-MFA script. This will process the MFA outputs into initial durations and modify the transcripts of your metadata.csv to be phonetic.

Again, this can take many arguments but all the defaults are good, you only need to specify the path of the config you're going to use (and the `--sample-rate` if it's something other than than 22050).
This is because the it needs the `hop_size` to calculate durations correctly. The `round y` is to enable rounding, which gives greater accuracy.

```
python examples/fastspeech2/mfa/postmfa.py --round y --yaml-path examples/fastspeech2/conf/fastspeech2.v1b.yaml
```
This will output the durations into a folder named `durations`

**Now run the usual preprocessing steps outlined in the main README, except that:**
In `tensorflow-tts-preprocess`, add `--trimlist trimlist.npy` as an argument. This is the trimming list output from postmfa.py

## After regular preprocessing

Normally, the features will be slightly higher in length than the extracted durations, which is why you must execute this. It compensates for the differences
Now, run the final duration postprocessing script. This one will drop the durations into the train and valid dump folders and match the lengths to the normalized features.

```
python examples/fastspeech2/mfa/postproc.py
```
This takes two arguments (`--dump-dir` and `--duration-path`) but the defaults are fine if you didn't change anything.

**After this, it should be ready to train**



