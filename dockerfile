FROM tensorflow/tensorflow:2.3.0rc0-gpu
RUN apt-get update
RUN apt-get install -y zsh tmux wget git
RUN pip install git+https://github.com/dathudeptrai/TensorflowTTS.git
RUN mkdir /workspace
WORKDIR /workspace
