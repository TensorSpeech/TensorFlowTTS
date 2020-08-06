from dataclasses import dataclass
import abc
import os
import json

from typing import Union


@dataclass
class ExamplePrepro(abc.ABC):
    root_dir: str
    symbols: list
    speakers_map = {}
    train_f_name: str = "train.txt"
    delimiter: str = "|"
    positions = {"file": 0, "text": 1, "speaker_name": 2}  # positions of file,text,speaker_name after split line
    f_extension: str = ".wav"
    extra_tokens = {"unk": "[UNK]", "pad": "[PAD]", "eos": "[EOS]", "bos": "[BOS]"}
    save_mapper: bool = False

    # extras
    items = []
    symbol_to_id = {}
    id_to_symbol = {}

    def __post_init__(self):
        self.create_items()
        self.create_speaker_map()
        self.reverse_speaker = {v: k for k, v in self.speakers_map.items()}
        self.create_symbols()
        if self.save_mapper:
            self.__save_mapper()

    def __getattr__(self, name: str) -> str:
        if "_id" in name:  # map extra token to id
            return self.symbol_to_id[self.extra_tokens[name.split("_id")[0]]]
        return self.extra_tokens[name]  # map extra token to value

    def create_speaker_map(self):
        """
        Create speaker map for dataset
        """
        sp_id = 0
        for i in self.items:
            speaker_name = i[-1]
            if speaker_name not in self.speakers_map:
                self.speakers_map[speaker_name] = sp_id
                sp_id += 1

    def get_speaker_id(self, name: str) -> int:
        return self.speakers_map[name]

    def get_speaker_name(self, speaker_id: int) -> str:
        return self.speakers_map[speaker_id]

    def create_symbols(self):
        self.symbols = self.symbols + list(self.extra_tokens.values())
        self.symbol_to_id = {s: i for i, s in enumerate(self.symbols)}
        self.id_to_symbol = {i: s for i, s in enumerate(self.symbols)}

    def create_items(self):
        """
        Method used to create items from training file
        items struct => text, wav_file_path, speaker_name
        """
        with open(os.path.join(self.root_dir, self.train_f_name), mode="r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(self.delimiter)
                wav_path = os.path.join(self.root_dir, parts[self.positions["file"]])
                wav_path = wav_path + self.f_extension if wav_path[-len(self.f_extension):] != self.f_extension else wav_path
                text = parts[self.positions["text"]]
                speaker_name = parts[self.positions["speaker_name"]]
                self.items.append([text, wav_path, speaker_name])

    def add_symbol(self, symbol: Union[str, list]):
        if isinstance(symbol, str):
            if symbol in self.symbol_to_id:
                return
            self.symbols.append(symbol)
            symbol_id = len(self.symbol_to_id)
            self.symbol_to_id[symbol] = symbol_id
            self.id_to_symbol[symbol_id] = symbol

        elif isinstance(symbol, list):
            for i in symbol:
                self.add_symbol(i)
        else:
            raise ValueError("A new_symbols must be a string or list of string.")

    @abc.abstractmethod
    def get_one_sample(self, item):
        """Get one sample from dataset items.
        Args:
            item: one item in Dataset items.
                Dataset items may include (raw_text, speaker_id, wav_path, ...)

        Returns:
            sample (dict): sample dictionary return all feature used for preprocessing later.
        """
        sample = {
            "raw_text": None,
            "text_ids": None,
            "audio": None,
            "utt_id": None,
            "speaker_name": None,
            "rate": None,
        }
        return sample

    @abc.abstractmethod
    def text_to_sequence(self, text: str):
        return []

    def convert_symbols_to_ids(self, symbols: Union[str, list]):
        sequence = []
        if isinstance(symbols, str):
            sequence.append(self._symbol_to_id[symbols])
            return sequence
        elif isinstance(symbols, list):
            for s in symbols:
                if isinstance(s, str):
                    sequence.append(self._symbol_to_id[s])
                else:
                    raise ValueError("All elements of symbols must be a string.")
        else:
            raise ValueError("A symbols must be a string or list of string.")

        return sequence

    def __save_mapper(self):
        """
        Save all needed mappers to file
        """
        with open(f"{self.root_dir}/mapper.json", "w") as f:
            full_mapper = {
                "symbol_to_id": self.symbol_to_id,
                "id_to_symbol": self.id_to_symbol,
                "speakers_map": self.speakers_map
            }
            json.dump(full_mapper, f)
