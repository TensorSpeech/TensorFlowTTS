import os
from dataclasses import dataclass
from pathlib import Path

import click
import numpy as np
import textgrid
import yaml
from tqdm import tqdm


@dataclass
class TxtGridParser:
    sample_rate: int
    multi_speaker: bool
    txt_grid_path: str
    hop_size: int
    durations_path: str
    dataset_path: str
    training_file: str = "train.txt"
    phones_mapper = {"sil": "SIL", "sp": "SIL", "spn": "SIL", "": "END"}
    """ '' -> is last token in every cases i encounter so u can change it for END but there is a safety check
        so it'll fail always when empty string isn't last char in ur dataset just chang it to silence then
    """
    sil_phones = set(phones_mapper.keys())

    def parse(self):
        speakers = (
            [
                i for i in os.listdir(self.txt_grid_path) if os.path.isdir(f"{self.txt_grid_path}/{i}")
            ]
            if self.multi_speaker
            else [])
        data = []

        if speakers:
            for speaker in speakers:
                file_list = os.listdir(f"{self.txt_grid_path}/{speaker}")
                self.parse_text_grid(file_list, data, speaker)
        else:
            file_list = os.listdir(self.txt_grid_path)
            self.parse_text_grid(file_list, data, "")

        with open(f"{self.dataset_path}/{self.training_file}", "w") as f:
            f.writelines(data)

    def parse_text_grid(self, file_list: list, data: list, speaker_name: str):
        print(f"\n parse: {len(file_list)} files, speaker name: {speaker_name} \n")
        for f_name in tqdm(file_list):
            text_grid = textgrid.TextGrid.fromFile(
                f"{self.txt_grid_path}/{speaker_name}/{f_name}"
            )
            pha = text_grid[1]
            durations = []
            phs = []
            for iterator, interval in enumerate(pha.intervals):
                mark = interval.mark

                if mark in self.sil_phones:
                    mark = self.phones_mapper[mark]
                    if mark == "END":
                        assert iterator == pha.intervals.__len__() - 1
                        # check if empty ph is always last example in your dataset if not fix it

                dur = interval.duration() * (self.sample_rate / self.hop_size)
                durations.append(round(dur))
                phs.append(mark)

            full_ph = " ".join(phs)

            assert full_ph.split(" ").__len__() == durations.__len__()  # safety check

            base_name = f_name.split(".TextGrid")[0]
            np.save(
                f"{self.durations_path}/{base_name}-durations.npy", np.array(durations)
            )
            data.append(f"{speaker_name}/{base_name}|{full_ph}|{speaker_name}\n")


@click.command()
@click.option(
    "--yaml_path", default="examples/fastspeech2_multispeaker/conf/fastspeech2.v1.yaml"
)
@click.option("--dataset_path", default="libritts", type=str, help="Dataset directory")
@click.option("--text_grid_path", default="mfa/parsed", type=str)
@click.option("--durations_path", default="libritts/durations")
@click.option("--sample_rate", default=24000, type=int)
@click.option("--multi_speakers", default=1, type=int, help="Use multi-speaker version")
@click.option("--train_file", default="train.txt")
def main(
    yaml_path: str,
    dataset_path: str,
    text_grid_path: str,
    durations_path: str,
    sample_rate: int,
    multi_speakers: int,
    train_file: str,
):

    with open(yaml_path) as file:
        attrs = yaml.load(file)
        hop_size = attrs["hop_size"]

    Path(durations_path).mkdir(parents=True, exist_ok=True)

    txt_grid_parser = TxtGridParser(
        sample_rate=sample_rate,
        multi_speaker=bool(multi_speakers),
        txt_grid_path=text_grid_path,
        hop_size=hop_size,
        durations_path=durations_path,
        training_file=train_file,
        dataset_path=dataset_path,
    )
    txt_grid_parser.parse()


if __name__ == "__main__":
    main()
