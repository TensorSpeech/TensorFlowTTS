import numpy as np
import os
from tqdm import tqdm
import click


@click.command()
@click.option("--base_path", default="dump")
@click.option("--trimmed_dur_path", default="dataset/trimmed-durations")
@click.option("--dur_path", default="dataset/durations")
@click.option("--use_norm", default="f")
def fix(base_path: str, dur_path: str, trimmed_dur_path: str, use_norm: str):
    for t in ["train", "valid"]:
        mfa_longer = []
        mfa_shorter = []
        big_diff = []
        not_fixed = []
        pre_path = f"{base_path}/{t}"
        os.makedirs(f"{pre_path}/fix_dur", exist_ok=True)

        print(f"FIXING {t}")
        for i in tqdm(os.listdir(f"{pre_path}/ids")):
            if use_norm == "t":
                mel = np.load(f"{pre_path}/norm-feats/{i.split('-')[0]}-norm-feats.npy")
            else:
                mel = np.load(f"{pre_path}/raw-feats/{i.split('-')[0]}-raw-feats.npy")

            try:
                dur = np.load(f"{trimmed_dur_path}/{i.split('-')[0]}-durations.npy")
            except:
                dur = np.load(f"{dur_path}/{i.split('-')[0]}-durations.npy")

            l_mel = len(mel)
            dur_s = np.sum(dur)
            cloned = np.array(dur, copy=True)
            diff = abs(l_mel - dur_s)

            if abs(l_mel - dur_s) > 30:  # more then 300 ms
                big_diff.append([i, abs(l_mel - dur_s)])

            if dur_s > l_mel:
                for j in range(1, len(dur) - 1):
                    if diff == 0:
                        break
                    dur_val = cloned[-j]

                    if dur_val >= diff:
                        cloned[-j] -= diff
                        diff -= dur_val
                        break
                    else:
                        cloned[-j] = 0
                        diff -= dur_val

                    if j == len(dur) - 2:
                        not_fixed.append(i)

                mfa_longer.append(abs(l_mel - dur_s))
            elif dur_s < l_mel:
                cloned[-1] += diff
                mfa_shorter.append(abs(l_mel - dur_s))

            np.save(f"{pre_path}/fix_dur/{i.split('-')[0]}-durations.npy", cloned)

        print(
            f"\n{t} stats: number of mfa with longer duration => {len(mfa_longer)} total diff => {sum(mfa_longer)}"
            f" mean diff => {sum(mfa_longer)/len(mfa_longer)}"
        )
        print(
            f"{t} stats: number of mfa with shorter duration => {len(mfa_shorter)} total diff => {sum(mfa_shorter)}"
            f" mean diff => {sum(mfa_shorter)/len(mfa_shorter) if len(mfa_shorter) > 0 else 0}"
        )
        print(
            f"{t} stats: number of files with a ''big'' duration diff => {len(big_diff)} if number>1 you should check it"
        )
        print(f"{t} stats: not fixed len => {len(not_fixed)}")


if __name__ == "__main__":
    fix()
