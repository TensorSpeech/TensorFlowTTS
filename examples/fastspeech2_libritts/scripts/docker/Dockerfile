FROM tensorflow/tensorflow:2.2.0-gpu
RUN apt-get update
RUN apt-get install -y zsh tmux wget git libsndfile1
ADD . /workspace/tts
WORKDIR /workspace/tts
RUN pip install .

