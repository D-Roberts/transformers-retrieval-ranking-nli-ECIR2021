FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

RUN apt update && \
    apt install -y bash \
    build-essential \
    wget \
    unzip \
    vim \
    git \
    curl \
    ca-certificates \
    python3 \
    python3-pip && \
    rm -rf /var/lib/apt/lists

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
    mkl \
    torch==1.5.1

ENV LC_ALL=C.UTF-8

RUN mkdir /mfactcheck
WORKDIR /mfactcheck

COPY ./requirements/requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

RUN mkdir -pv scripts
RUN mkdir -pv src

COPY scripts scripts
COPY src src

ENV PYTHONPATH "${PYTHONPATH}:src"

CMD ["/bin/bash"]