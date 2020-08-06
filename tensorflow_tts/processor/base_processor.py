# -*- coding: utf-8 -*-
# Copyright 2020 The TensorFlowTTS Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Abstract class for processor"""


import abc


class BaseProcessor(metaclass=abc.ABCMeta):
    """Base Processor class for all Processor."""


    SPECIAL_TOKENS_ATTRIBUTES = [
        "unk",
        "pad",
        "eos",
        "bos",
    ]

    def __init__(
        self,
        items,
        symbols,
        unk="[UNK]",
        pad="[PAD]",
        eos="[EOS]",
        bos="[BOS]",
        **kwargs
    ):
        super().__init__(**kwargs)
        self._unk = unk
        self._pad = pad
        self._eos = eos
        self._bos = bos

        # Update SPECIAL_TOKENS.
        for key, value in kwargs.items():
            if key in self.SPECIAL_TOKENS_ATTRIBUTES:
                if isinstance(value, str):
                    setattr(self, key, value)
                else:
                    raise TypeError(
                        "special token {} has to be str but got: {}".format(key, type(value))
                    )

        self._items = None
        self._symbols = None

        self.items = items
        self.symbols = symbols

        # update symbols with special tokens
        # to keep backward compatibility, pad always a first symbols
        # and eos always after self_symbols.
        self.symbols = [self.pad] + self._symbols + [self.eos] + [self.bos] + [self.unk]
        
        # symbols mapping
        self._symbol_to_id = {s: i for i, s in enumerate(self.symbols)}
        self._id_to_symbol = {i: s for i, s in enumerate(self.symbols)}

    def add_symbols(self, new_symbols):
        """Add new symbols to the end of self._symbols."""
        if isinstance(new_symbols, str):
            self._symbols = self._symbols + [new_symbols]
        elif isinstance(new_symbols, list):
            for s in new_symbols:
                if isinstance(s, str):
                    self._symbols = self._symbols + [s]
                else:
                    raise ValueError("All elements of new_symbols must be a str.")
        else:
            raise ValueError("A new_symbols must be a string or list of string.")

    @property
    def unk(self):
        """Unkow character/phoneme."""
        return str(self._unk)

    @property
    def pad(self):
        """Padding."""
        return str(self._pad)

    @property
    def eos(self):
        """end of sentence."""
        return str(self._eos)

    @property
    def bos(self):
        """begin of sentence."""
        return str(self._bos)

    @property
    def items(self):
        return self._items

    @items.setter
    def items(self, value):
        self._items = value

    @property
    def symbols(self):
        return self._symbols
    
    @symbols.setter
    def symbols(self, value):
        self._symbols = value

    @property
    def symbol_to_id(self):
        return self._symbol_to_id

    @property
    def id_to_symbol(self):
        return self._id_to_symbol

    @unk.setter
    def unk(self, value):
        self._unk = value

    @pad.setter
    def pad(self, value):
        self._pad = value

    @eos.setter
    def eos(self, value):
        self._eos = value

    @bos.setter
    def bos(self, value):
        self._bos = value

    @property
    def unk_id(self):
        if self._unk is None:
            return None
        return self.convert_symbols_to_ids(self.eos)

    @property
    def pad_id(self):
        if self._pad is None:
            return None
        return self.convert_symbols_to_ids(self.pad)

    @property
    def eos_id(self):
        if self._eos is None:
            return None
        return self.convert_symbols_to_ids(self.eos)

    @property
    def bos_id(self):
        if self._bos is None:
            return None
        return self.convert_symbols_to_ids(self.bos)
    
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
    def text_to_sequence(self, text):
        return []

    def convert_symbols_to_ids(self, symbols):
        sequence = []
        if isinstance(symbols, str):
            sequence.append(self._symbol_to_id[symbols])
            return sequence
        elif isinstance(symbols, list):
            for s in symbols:
                if isinstance(s, str):
                    sequence.append(self._symbol_to_id[s])
                else:
                    raise ValueError("All elements of symbols must be a str.")
        else:
            raise ValueError("A symbols must be a string or list of string.")
        
        return sequence


