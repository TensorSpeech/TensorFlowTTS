FROM tensorflow/tensorflow:2.2.0-gpu
RUN apt-get update
RUN apt-get install -y zsh tmux wget git
RUN pip install git+github.com/TensorSpeech/TensorflowTTS.git
RUN mkdir /workspace
WORKDIR /workspace
