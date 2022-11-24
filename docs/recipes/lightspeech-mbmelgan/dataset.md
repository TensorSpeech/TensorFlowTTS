# Dataset

## Folder Structure

The structure of the training files is fairly simple and straightforward:

```
{YOUR_DATASET}/
├── {SPEAKER-1}/
│   ├── {SPEAKER-1}_{UTTERANCE-0}.lab
│   ├── {SPEAKER-1}_{UTTERANCE-0}.wav
│   ├── {SPEAKER-1}_{UTTERANCE-1}.lab
│   ├── {SPEAKER-1}_{UTTERANCE-1}.wav
│   ├── ...
├── {SPEAKER-2}/
│   ├── {SPEAKER-2}_{UTTERANCE-0}.lab
│   ├── {SPEAKER-2}_{UTTERANCE-0}.wav
│   ├── ...
├── ...
```

A few key things to note here:

- Each speaker has its own subfolder within the root dataset folder.
- The filenames in the speaker subfolders follow the convention of `{SPEAKER-#}_{UTTERANCE-#}`. It is important that they are delimited by an underscore (`_`), so make sure that there is no `_` within the speaker name and within the utterance ID. Use dashes `-` instead within them instead.
- Audios are in `wav` format and transcripts are of `lab` format (same content as you expect from a `txt` file; nothing fancy about it). The reason we use `lab` is simply to facilitate Montreal Forced Aligner training later.

### Example

In the root directory `en-bookbot`, there are three speakers: `en-AU-Zak`, `en-UK-Thalia`, and `en-US-Madison`. The struture of the files are as follows:

```
en-bookbot/
├── en-AU-Zak/
│   ├── en-AU-Zak_0.lab
│   ├── en-AU-Zak_0.wav
│   ├── en-AU-Zak_1.lab
│   ├── en-AU-Zak_1.wav
│   ├── ...
├── en-UK-Thalia/
│   ├── en-UK-Thalia_0.lab
│   ├── en-UK-Thalia_0.wav
│   ├── ...
└── en-US-Madison/
    ├── en-US-Madison_0.lab
    ├── en-US-Madison_0.wav
    ├── ...
```

## Lexicon

Another required component for training a Montreal Forced Aligner is a lexicon file (usually named `lexicon.txt`). A lexicon simply maps words (graphemes) to phonemes, i.e. a pronunciation dictionary. Later, these phonemes will be aligned to segments of audio that they correpond to, and its duration will be learned by the forced aligner.

There are many available lexicons out there, such as ones provided by [Montreal Forced Aligner](https://mfa-models.readthedocs.io/en/latest/dictionary/index.html) and [Open Dict Data](https://github.com/open-dict-data/ipa-dict). You can either find other pre-existing lexicons, or create your own. Otherwise, another option would to treat each grapheme character as proxy phonemes as done by [Meyer et al. (2022)](https://arxiv.org/abs/2207.03546) where a lexicon is unavailable in certain languages:

> Two languages (ewe and yor) were aligned via forced alignment from scratch. Using only the found audio and transcripts (i.e., without a pre-trained acoustic model), an acoustic model was trained and the data aligned with the Montreal Forced Aligner. Graphemes were used as a proxy for phonemes in place of G2P data.

In any case, the lexicon file should consist of `tab`-delimited word-phoneme pairs, which looks like the following:

```
what    w ˈʌ t
is    ˈɪ z
biology    b aɪ ˈɑ l ə d͡ʒ i
?    ?
the    ð ə
study    s t ˈʌ d i
of    ə v
living    l ˈɪ v ɪ ŋ
things    θ ˈɪ ŋ z
.    .
from    f ɹ ˈʌ m
...
```

Another example would be:

```
:    :
,    ,
.    .
!    !
?    ?
;    ;
a    a
b    b e
c    tʃ e
d    d e
abad    a b a d
abadi    a b a d i
abadiah    a b a d i a h
abadikan    a b a d i k a n
...
```

There are a few key things to note as well:

- The lexicon should cover all words in the audio corpus -- no out-of-vocabulary words should be present during the training of the aligner later. Otherwise, this will result in unknown tokens and will likely disrupt the training and duration parsing processes.
- Include all the punctuations you would want to have in the model later. For instance, I would usually keep `. , : ; ? !` because they might imply different pause durations and/or intonations. Every other punctuations not in the lexicon will be stripped during the alignment process.
- Individual phonemes should be separated with whitespaces, e.g. `a` is its own phoneme unit, and `tʃ` is also considered as another single phoneme unit despite having 2 characters.

Structuring your dataset, preprocessing, and lexicon preparation are arguably the most complicated part of training these kinds of models. But once we're over this step, everything else should be quite easy to follow.