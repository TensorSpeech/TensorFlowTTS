"""Setup Tensorflow TTS libarary."""

import os
import sys
from distutils.version import LooseVersion

import pip
from setuptools import find_packages, setup

if LooseVersion(sys.version) < LooseVersion("3.6"):
    raise RuntimeError(
        "TensorFlow TTS requires python >= 3.6, "
        "but your Python version is {}".format(sys.version)
    )

if LooseVersion(pip.__version__) < LooseVersion("19"):
    raise RuntimeError(
        "pip>=19.0.0 is required, but your pip version is {}. "
        'Try again after "pip install -U pip"'.format(pip.__version__)
    )

# TODO(@dathudeptrai) update requirement if needed.
requirements = {
    "install": [
        "tensorflow-gpu==2.7.0",
        "tensorflow-addons>=0.10.0",
        "setuptools>=38.5.1",
        "huggingface_hub==0.0.8",
        "librosa>=0.7.0",
        "soundfile>=0.10.2",
        "matplotlib>=3.1.0",
        "PyYAML>=3.12",
        "tqdm>=4.26.1",
        "h5py>=2.10.0",
        "unidecode>=1.1.1",
        "inflect>=4.1.0",
        "scikit-learn>=0.22.0",
        "pyworld>=0.2.10",
        "numba<=0.48",  # Fix No module named "numba.decorators"
        "jamo>=0.4.1",
        "pypinyin",
        "g2pM",
        "textgrid",
        "click",
        "g2p_en",
        "dataclasses",
    ],
    "setup": ["numpy", "pytest-runner",],
    "test": [
        "pytest>=3.3.0",
        "hacking>=1.1.0",
    ],
}

# TODO(@dathudeptrai) update console_scripts.
entry_points = {
    "console_scripts": [
        "tensorflow-tts-preprocess=tensorflow_tts.bin.preprocess:preprocess",
        "tensorflow-tts-compute-statistics=tensorflow_tts.bin.preprocess:compute_statistics",
        "tensorflow-tts-normalize=tensorflow_tts.bin.preprocess:normalize",
    ]
}

install_requires = requirements["install"]
setup_requires = requirements["setup"]
tests_require = requirements["test"]
extras_require = {
    k: v for k, v in requirements.items() if k not in ["install", "setup"]
}

dirname = os.path.dirname(__file__)
setup(
    name="TensorFlowTTS",
    version="0.0",
    url="https://github.com/tensorspeech/TensorFlowTTS",
    author="Minh Nguyen Quan Anh, Alejandro Miguel Velasquez, Dawid Kobus, Eren Gölge, Kuan Chen, Takuya Ebata, Trinh Le Quang, Yunchao He",
    author_email="nguyenquananhminh@gmail.com",
    description="TensorFlowTTS: Real-Time State-of-the-art Speech Synthesis for TensorFlow 2",
    long_description=open(os.path.join(dirname, "README.md"), encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    packages=find_packages(include=["tensorflow_tts*"]),
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    extras_require=extras_require,
    entry_points=entry_points,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
