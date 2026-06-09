FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends wget ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# install micromamba
RUN wget -qO /usr/local/bin/micromamba \
    "https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64" && \
    chmod +x /usr/local/bin/micromamba

ENV MAMBA_ROOT_PREFIX=/opt/conda

# install conda packages (openbabel must come from conda-forge, not pip)
COPY environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml && \
    micromamba clean --all --yes

ENV PATH=/opt/conda/bin:${PATH}

# install paddlepaddle-gpu (CUDA 11.8) and PGL
# if your host CUDA version differs, change cu118 below to match (e.g. cu117, cu120)
RUN pip install --no-cache-dir \
        paddlepaddle-gpu==2.5.2 \
        -i https://www.paddlepaddle.org.cn/packages/stable/cu118/ && \
    pip install --no-cache-dir "pgl>=2.1.4"

RUN python -c "import openbabel; import paddle; import pgl; print('env ok')"

WORKDIR /work
COPY . /work
