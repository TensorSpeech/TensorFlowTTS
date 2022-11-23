# Dataset Structure

The structure of the training files is fairly simple and straightforward:

```
|- {YOUR_DATASET}/
|   |- {SPEAKER-1}/
|   |-  |- {SPEAKER-1}_{UTTERANCE-0}.lab
|   |-  |- {SPEAKER-1}_{UTTERANCE-0}.wav
|   |-  |- {SPEAKER-1}_{UTTERANCE-1}.lab
|   |-  |- {SPEAKER-1}_{UTTERANCE-1}.wav
|   |-  |- ...
|   |- {SPEAKER-2}/
|   |-  |- {SPEAKER-2}_{UTTERANCE-0}.lab
|   |-  |- {SPEAKER-2}_{UTTERANCE-0}.wav
|   |-  |- ...
|   |- ...
```

A few key things to note here:

- Each speaker has its own subfolder within the root dataset folder.
- The filenames in the speaker subfolders follow the convention of `{SPEAKER-#}_{UTTERANCE#}`. It is important that they are delimited by an underscore (`_`), so make sure that there is no `_` within the speaker name and within the utterance ID. Use dashes `-` instead within them instead.
- Audios are in `wav` format and transcripts are of `lab` format (same content as you expect from a `txt` file; nothing fancy about it). The reason we use `lab` is simply to facilitate Montreal Forced Aligner training later.