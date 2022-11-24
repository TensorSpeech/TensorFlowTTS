# Installation

## Local Machines

I highly recommend installing TensorFlowTTS (and TensorFlow) on a designated Conda environment. I personally prefer Miniconda over Anaconda, but either one works. To begin with, follow [this guide](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) to install Conda, and then create a new Python 3.9 environment, which I will call `tensorflow`.

```sh
conda create -n tensorflow python=3.9
conda activate tensorflow
```

In the new environment, I will install TensorFlow v2.3.1 which I have found to work for training and inference later. You can install it via `pip`.

```sh
pip install tensorflow==2.3.1
```

Afterwards, clone the forked repository and install the library plus all of its requirements.

```sh
git clone https://github.com/w11wo/TensorFlowTTS.git
cd TensorFlowTTS
pip install .
```

## Google Cloud Virtual Machines

Installing TensorFlowTTS on a Google Cloud VM is similar to installing on a local machine. To make things easier, Google has provided us with a list of pre-built VM images that comes with TensorFlow and support for GPUs. I would go for the image: **Debian 10 based Deep Learning VM for TensorFlow Enterprise 2.6 with CUDA 11.0**.

Because the image already has TensorFlow installed, we just need to install the main library like the steps above

```sh
git clone https://github.com/w11wo/TensorFlowTTS.git
cd TensorFlowTTS
pip install .
```

For some reason, there will be a bug involving Numba, which we can easily solve by upgrading NumPy to the latest version

```sh
pip install -U numpy
```

And also install `libsndfile1` via `apt`

```sh
sudo apt-get install libsndfile1
```

## Checking for a Successful Install

A way to check if your installation is correct is by importing the library through Python. We can do so through command line.

```sh
python -c "import tensorflow_tts"
```

If no errors are raised, then we should be good to go!