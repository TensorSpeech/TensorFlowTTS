FROM tensorflow/tensorflow:2.6.0-gpu
RUN apt-get update
RUN apt-get install -y zsh tmux wget git libsndfile1
RUN pip install ipython && \
    pip install git+https://github.com/TensorSpeech/TensorflowTTS.git && \
    pip install git+https://github.com/repodiac/german_transliterate.git#egg=german_transliterate
RUN mkdir /workspace
WORKDIR /workspace
