# -*- coding: utf-8 -*-
# Copyright 2020 TensorFlowTTS Team.
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
"""Runing mfa to extract textgrids."""

from subprocess import call
from pathlib import Path

import click
import os


@click.command()
@click.option("--mfa_path", default=os.path.join('mfa', 'montreal-forced-aligner', 'bin', 'mfa_align'))
@click.option("--corpus_directory", default="libritts")
@click.option("--lexicon", default=os.path.join('mfa', 'lexicon', 'librispeech-lexicon.txt'))
@click.option("--acoustic_model_path", default=os.path.join('mfa', 'montreal-forced-aligner', 'pretrained_models', 'english.zip'))
@click.option("--output_directory", default=os.path.join('mfa', 'parsed'))
@click.option("--jobs", default="8")
def run_mfa(
    mfa_path: str,
    corpus_directory: str,
    lexicon: str,
    acoustic_model_path: str,
    output_directory: str,
    jobs: str,
):
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    call(
        [
            f".{os.path.sep}{mfa_path}",
            corpus_directory,
            lexicon,
            acoustic_model_path,
            output_directory,
            f"-j {jobs}"
         ]
    )


if __name__ == "__main__":
    run_mfa()
