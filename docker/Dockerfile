FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime
WORKDIR /workspace
RUN apt-get update && apt-get install -y --no-install-recommends git \
    build-essential \
    ca-certificates \
    ccache \
    cmake \
    curl \
    git \
    libsndfile1
RUN /opt/conda/bin/pip install git+git://github.com/pytorch/fairseq.git@b8ea8a9b72c82192da07e3377adf4ebbde16716d
RUN /opt/conda/bin/pip install pytorch-lightning==1.4.5 audlib transformers==4.10.0 soundfile matplotlib seaborn
RUN /opt/conda/bin/pip install librosa
RUN conda install -c conda-forge faiss-gpu 

WORKDIR /workspace/project
