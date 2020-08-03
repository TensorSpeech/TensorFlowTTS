# MFA based extraction for Fast speech

## Prepare
Everything is done from main repo folder so TensorflowTTS/

* bash examples/mfa_extraction/scripts/prepare_mfa.sh
* python examples/mfa_extraction/run_mfa.py --corpus_directory=<your dataset path>
   
   (corpus_directory should be splited based on speakers example => dataset/speaker_1/001.wav dataset/speaker_1/001.txt)
   
* Add your own dataset parser based on prepro/experiment/example_dataset.py to prepro/preprocess.py

* Run prepro/preprocess.py first remove comments from end of the file you need to change end of file too looks like this also add parameters needed(rootdir, outdir and config)
```python
if __name__ == "__main__":  # TODO change this later
    preprocess()
    # normalize()
```
* Run same script but change option to normalize() 
```python
if __name__ == "__main__":  # TODO change this later
    # preprocess()
    normalize()
```
* Run examples/mfa_extraction/fix_mismatch.py --base_path=<your preprocess outdir location> 
--trimmed_dur_path=<trimmed durations directory> --dur_path=<durations directory>

## Comments
step 3 and 4 will be changed I'm considering move everything/dont install tensorflow tts folder as its slow development drastically (few times i didnt remember to reinstall package or tftts got installed multiple times so i need to uninstall it multiple times in anacoda env it also make debbuging A LOT harder) 

## Problems with MFA extraction
Looks like MFA have problems with trimmed files it works better (in my experiments) with ~100ms of silence at start and end
Short files can not get false positive only silence extraction (LibriTTS example) so i would get only samples >2s