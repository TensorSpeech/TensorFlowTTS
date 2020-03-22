"""Setup Tensorflow TTS libarary."""

import os
import pip
import sys

from distutils.version import LooseVersion
from setuptools import find_packages
from setuptools import setup

if LooseVersion(sys.version) < LooseVersion("3.6"):
    raise RuntimeError(
        "Tensorflow TTS requires python >= 3.6, "
        "but your Python version is {}".format(sys.version)
    )

if LooseVersion(pip.__version__) < LooseVersion("19"):
    raise RuntimeError(
        "pip>=19.0.0 is required, but your pip version is {}. "
        "Try again after \"pip install -U pip\"".format(pip.__version__)
    )

# TODO(@dathudeptrai) update requirement if needed.
requirements = {
    "install": [
        "tensorflow-gpu>=2.1.0",
        "setuptools>=38.5.1",
        "librosa>=0.7.0",
        "soundfile>=0.10.2",
        "matplotlib>=3.1.0",
        "PyYAML>=3.12",
        "tqdm>=4.26.1",
        "h5py>=2.10.0",
    ],
    "setup": [
        "numpy",
        "pytest-runner",
    ],
    "test": [
        "pytest>=3.3.0",
        "hacking>=1.1.0",
        "flake8>=3.7.8",
        "flake8-docstrings>=1.3.1",
    ]
}

# TODO(@dathudeptrai) update console_scripts.
entry_points = {
    "console_scripts": [
        # TODO
    ]
}

install_requires = requirements["install"]
setup_requires = requirements["setup"]
tests_require = requirements["test"]
extras_require = {k: v for k, v in requirements.items()
                  if k not in ["install", "setup"]}

dirname = os.path.dirname(__file__)
setup(name="tensorflow_tts",
      version="0.0.0",
      url="https://github.com/dathudeptrai/TensorflowTTS",
      author="Minh Nguyen Quan Anh, Trinh Le Quang, Quoc Van Huu",
      author_email="nguyenquananhminh@gmail.com",
      description="Deep learning for Text to Speech implementation with Tensorflow",
      long_description=open(os.path.join(dirname, "README.md"),
                            encoding="utf-8").read(),
      long_description_content_type="text/markdown",
      license="MIT License",
      packages=find_packages(include=["tensorflow_tts*"]),
      install_requires=install_requires,
      setup_requires=setup_requires,
      tests_require=tests_require,
      extras_require=extras_require,
      entry_points=entry_points,
      classifiers=[
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "Intended Audience :: Science/Research",
          "Operating System :: POSIX :: Linux",
          "License :: OSI Approved :: MIT License",
          "Topic :: Software Development :: Libraries :: Python Modules"
      ],
)