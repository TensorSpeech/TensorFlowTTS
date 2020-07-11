from tqdm import tqdm
import os
import argparse
def safemkdir(dirn):
  if not os.path.isdir(dirn):
    os.mkdir(dirn)

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for MFA align")
    parser.add_argument(
        "--dataset-path",
        default="datasets",
        type=str,
        help="Path of LJSpeech or like dataset",
    )
    parser.add_argument(
        "--out-path",
        default="TextGrid",
        type=str,
        help="Directory to create for TextGrid output",
    )
    args = parser.parse_args()
    metadpath = args.dataset_path + "/metadata.csv"
    datapath = args.dataset_path + "/wavs"

    safemkdir(args.out_path)
    print("Preparing dataset for MFA...")
    with open(metadpath,"r") as f:
      for mli in tqdm(f.readlines()):
        lisplit = mli.strip().split("|")

        rawpath = lisplit[0]
        transcr = lisplit[2]


        pfileout = datapath + "/" + rawpath + ".lab"
        fout = open(pfileout,"w")
        fout.write(transcr + "\n")
        fout.close()

if __name__ == "__main__":
    main()

