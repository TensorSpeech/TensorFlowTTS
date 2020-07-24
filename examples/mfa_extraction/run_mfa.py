from subprocess import call
from pathlib import Path

import click


@click.command()
@click.option("--mfa_path", default="mfa/montreal-forced-aligner/bin/mfa_align")
@click.option("--corpus_directory", default="dataset")
@click.option("--lexicon", default="mfa/lexicon/librispeech-lexicon.txt")
@click.option("--acoustic_model_path", default="mfa/montreal-forced-aligner/pretrained_models/english.zip")
@click.option("--output_directory", default="mfa/parsed")
@click.option("--jobs", default="3")
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
            f"./{mfa_path}",
            corpus_directory,
            lexicon,
            acoustic_model_path,
            output_directory,
            f"-j {jobs}"
         ]
    )


if __name__ == "__main__":
    run_mfa()
