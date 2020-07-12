import textgrid
import yaml
import os
import numpy as np
from tqdm import tqdm
import argparse

def safemkdir(dirn):
  if not os.path.isdir(dirn):
    os.mkdir(dirn)
    
def main():
    parser = argparse.ArgumentParser(description="Read durations from MFA and assign")
    parser.add_argument(
        "--yaml-path",
        default=None,
        type=str,
        help="Path of FastSpeech2 config. Will be used for extracting the hop_size",
    )
    parser.add_argument(
        "--dataset-path",
        default="datasets",
        type=str,
        help="Dataset directory",
    )
    parser.add_argument(
       "--textgrid-path",
       default="TextGrids",
        type=str,
        help="Directory where the TextGrid output from MFA is stored",
    )
    parser.add_argument(
       "--duration-path",
       default="durations",
        type=str,
        help="Directory where the duration output will be stored",
    )
    parser.add_argument(
       "--sample-rate",
       default=22050,
        type=int,
        help="Sample rate of source audio",
    )
    parser.add_argument(
       "--trimlistname",
       default="trimlist",
        type=str,
        help="Name of trimlist output",
    )
    args = parser.parse_args()
    hopsz = 256
    sarate = args.sample_rate
    
    yapath = args.yaml_path
    inmetadpath = args.dataset_path + "/metadata.csv"
    wavspath = args.dataset_path + "/wavs"
    txgridpath = args.textgrid_path
    
    tgrids = os.listdir(txgridpath)

    with open(yapath) as file:
        attrs = yaml.load(file)
        hopsz = attrs["hop_size"]

    durationpath = args.duration_path
    safemkdir(durationpath)
    sil_phones = ['sil', 'sp', 'spn', '']
    metafile = open(inmetadpath,"w")
    print("Reading TextGrids...")

    trimidlist = []
    trimdurlist = []
    for tgp in tqdm(tgrids):
      if not os.path.isfile(txgridpath + "/" + tgp):
        print("Could not find " + tgp)
        if len(wavspath) > 1:
          wavefn = wavspath + "/" + tgp.replace(".TextGrid",".wav")
          if os.path.isfile(wavefn):
            print("Deleting " + wavefn)
            os.remove(wavefn)
        continue

      tg = textgrid.TextGrid.fromFile(txgridpath + "/" + tgp)
      pha = tg[1]
      durations = []
      totdursecs = 0.0
      phs = "{"
      for interval in pha.intervals:
        mark = interval.mark
        if mark in sil_phones:
          mark = "SIL"
        dur = interval.duration()*(sarate/hopsz)
        durations.append(int(dur))
        phs += mark + " "
        totdursecs += interval.duration()
      phs += "END"
      durations.append(0)
      phs += "}"
      phs = phs.replace(" }","}")
      trimidlist.append(tgp.replace(".TextGrid",""))
      trimdurlist.append(totdursecs)

      
      
      np.save(durationpath + "/" + tgp.replace(".TextGrid","-durations"),np.array(durations))
      metafile.write(tgp.replace(".TextGrid","") + "|" + phs + "|" + phs + "\n")
      


    metafile.close()
    np.save(args.trimlistname,np.array([np.array(trimidlist),np.array(trimdurlist)]))
  
  

if __name__ == "__main__":
    main()


