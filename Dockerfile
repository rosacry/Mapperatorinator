FROM nvcr.io/nvidia/pytorch:24.07-py3

RUN apt-get -y update && apt-get -y upgrade && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
RUN pip install accelerate pydub nnAudio PyYAML transformers hydra-core tensorboard slider==0.8.1 torch_tb_profiler wandb ninja
RUN MAX_JOBS=4 pip install flash-attn --no-build-isolation

# Modify .bashrc to include the custom prompt
RUN echo 'if [ -f /.dockerenv ]; then export PS1="(docker) $PS1"; fi' >> /root/.bashrc
