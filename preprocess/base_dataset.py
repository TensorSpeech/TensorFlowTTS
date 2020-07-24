import abc
from dataclasses import dataclass
from abc import abstractmethod
import os


@dataclass
class BaseDataset(abc.ABC):
    root_path: str
    speakers_map: dict
    train_f_name: str = "train.txt"
    delimiter: str = "|"
    positions = {"file": 0, "text": 1, "speaker_name": 2}  # positions of file,text,speaker_name after split line
    f_extension: str = ".wav"
    items = []

    def __post_init__(self):
        self.create_items()
        self.reverse_speaker = {v: k for k, v in self.speakers_map.items()}

    def get_speaker_id(self, name: str) -> int:
        return self.speakers_map[name]

    def get_speaker_name(self, speaker_id: int) -> str:
        return self.speakers_map[speaker_id]

    def create_items(self):
        with open(os.path.join(self.root_path, self.train_f_name), mode="r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(self.delimiter)
                wav_path = os.path.join(self.root_path, parts[self.positions["file"]])
                wav_path = wav_path + self.f_extension if wav_path[-len(self.f_extension):] != self.f_extension else wav_path
                text = parts[self.positions["text"]]
                speaker_name = parts[self.positions["speaker_name"]]
                self.items.append([text, wav_path, speaker_name])

    @abstractmethod
    def get_one_sample(self, idx: int):
        raise NotImplementedError

    @abstractmethod
    def text_to_sequence(self, text: str):
        raise NotImplementedError
