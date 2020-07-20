import os
import shutil
from tqdm import tqdm
import argparse
import numpy as np
def safemkdir(dirn):
  if not os.path.isdir(dirn):
    os.mkdir(dirn)

from pathlib import Path
def main():
    parser = argparse.ArgumentParser(description="Postprocess MFA alignments including length and split matching")
    parser.add_argument(
        "--dump-dir",
        default="dump",
        type=str,
        help="Path of dump directory",
    )
    parser.add_argument(
        "--duration-path",
        default="durations",
        type=str,
        help="Directory of durations output",
    )
    args = parser.parse_args()
    origdurpath = args.duration_path
    valpath = args.dump_dir + "/valid"
    trainpath = args.dump_dir + "/train"
    durpath = trainpath + "/durations"
    
    print("Step 1/1 of stage 1/2: Move durations to dump and split")
    
    print("Moved all durations to train dir")
    shutil.move(origdurpath,durpath)
    outvalpath = valpath + "/durations/"
    idvalpaths = valpath + "/ids"
    safemkdir(outvalpath)
    
    print("Find validation and move duration from train to valid dir")
    for fn in tqdm(os.listdir(idvalpaths)):
      mvid = fn.replace("-ids","-durations")
      shutil.move(durpath + "/" + mvid,outvalpath)
      
    print("Step 1/2: Match duration to sets")
    sets = ["train", "valid"]
    afterdel = list()
    
    for set in sets:
        print("Matching to " + set + " set.")
        durlog = open("durcomp_" + set + ".txt","w")
        dumpstage = args.dump_dir + "/" + set
        featpath = dumpstage + "/norm-feats"
        zdurpath = dumpstage + "/durations"

        idsq = os.listdir(featpath)
        for idf in tqdm(idsq):
          melz = np.load(featpath + "/" + idf)
          durload = zdurpath + "/" + idf.replace("-norm-feats","-durations")
          duraz = np.load(durload)

          mellen = len(melz)
          durlen = np.sum(duraz)

          if mellen > durlen:
            duraz[-1] += mellen - durlen
          else:
            if durlen > mellen:
              diff = durlen - mellen
              found = False
              for r in reversed(range(len(duraz) - 1)):
                if duraz[r] >= diff:
                  duraz[r] -= diff
                  found = True
                  break
              if not found:
                trueid = idf.replace("norm-feats.npy","")
                print("Not found for " + trueid + " adding to list")
                afterdel.append(trueid)





          durlog.write(str(mellen) + "|" + str(durlen) + "\n")
          np.save(durload,duraz)
        durlog.close()
    
    for trueid in afterdel:
      searchterm = trueid + "*"
      print(searchterm)
      for path in Path(args.dump_dir).rglob(searchterm):
        print("Deleting: " + path.name)
        os.remove(path.as_posix())
     
     

    
      
    
    
         


      
if __name__ == "__main__":
    main()

