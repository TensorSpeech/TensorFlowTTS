import pytest
from tensorflow_tts.processor.base_processor import BaseProcessor, DataProcessorError
import string
from dataclasses import dataclass
from shutil import copyfile


@dataclass
class LJ(BaseProcessor):
    def get_one_sample(self, item):
        sample = {
            "raw_text": None,
            "text_ids": None,
            "audio": None,
            "utt_id": None,
            "speaker_name": None,
            "rate": None,
        }
        return sample

    def text_to_sequence(self, text):
        return ["0"]

    def setup_eos_token(self):
        return None


@pytest.fixture
def processor(tmpdir):
    copyfile("test/files/train.txt", f"{tmpdir}/train.txt")
    processor = LJ(data_dir=tmpdir, symbols=list(string.ascii_lowercase))
    return processor


@pytest.fixture
def mapper_processor(tmpdir):
    copyfile("test/files/train.txt", f"{tmpdir}/train.txt")
    copyfile("test/files/mapper.json", f"{tmpdir}/mapper.json")
    processor = LJ(data_dir=tmpdir, loaded_mapper_path=f"{tmpdir}/mapper.json")
    return processor


def test_items_creation(processor):
    # Check text
    assert processor.items[0][0] == "in fact its just a test."
    assert processor.items[1][0] == "in fact its just a speaker number one."

    # Check path
    assert processor.items[0][1].split("/")[-1] == "libri1.wav"
    assert processor.items[1][1].split("/")[-1] == "libri2.wav"

    # Check speaker name
    assert processor.items[0][2] == "One"
    assert processor.items[1][2] == "Two"


def test_mapper(processor):
    # check symbol to id mapper
    assert processor.symbol_to_id["a"] == 0

    # check id to symbol mapper
    assert processor.id_to_symbol[0] == "a"

    # check speaker mapper
    assert processor.speakers_map["One"] == 0
    assert processor.speakers_map["Two"] == 1


def test_adding_symbols(processor):
    # check symbol to id mapper
    assert processor.symbol_to_id["a"] == 0

    # check id to symbol mapper
    assert processor.id_to_symbol[0] == "a"

    old_processor_len = len(processor.symbols)

    # Test adding new symbol
    processor.add_symbol("O_O")

    assert processor.symbol_to_id["a"] == 0
    assert (
        processor.symbol_to_id["O_O"] == len(processor.symbols) - 1
    )  # new symbol should have last id

    assert processor.id_to_symbol[0] == "a"
    assert processor.id_to_symbol[len(processor.symbols) - 1] == "O_O"

    assert old_processor_len == len(processor.symbols) - 1


def test_loading_mapper(mapper_processor):
    assert mapper_processor.symbol_to_id["a"] == 0
    assert mapper_processor.symbol_to_id["@ph"] == 2

    assert mapper_processor.speakers_map["test_one"] == 0
    assert mapper_processor.speakers_map["test_two"] == 1

    assert mapper_processor.id_to_symbol[0] == "a"
    assert mapper_processor.id_to_symbol[2] == "@ph"

    # Test failed creation
    with pytest.raises(DataProcessorError):
        failed = LJ(data_dir="test/files")
