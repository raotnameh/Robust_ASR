FROM nvidia/cuda:10.1-devel-ubuntu18.04

ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

WORKDIR /workspace/

# install basics
RUN apt-get update -y
RUN apt-get install -y git curl ca-certificates bzip2 cmake tree htop bmon iotop sox libsox-dev libsox-fmt-all vim
RUN apt-get install -y python3-pip
RUN apt-get install -y python3-setuptools

#install pytorch
RUN pip3 install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# install warp-CTC
ENV CUDA_HOME=/usr/local/cuda
RUN git clone https://github.com/SeanNaren/warp-ctc.git
RUN cd warp-ctc; mkdir build; cd build; cmake ..; make
RUN cd warp-ctc; cd pytorch_binding; python3 setup.py install

# install ctcdecode
RUN git clone --recursive https://github.com/parlance/ctcdecode.git
RUN cd ctcdecode; pip3 install .

# install apex
RUN git clone --recursive https://github.com/NVIDIA/apex.git
RUN cd apex; pip3 install .

#install llvm dependencies
RUN apt install -y llvm-10
RUN cd /usr/bin; ln -s llvm-config-10 llvm-config;cd /workspace
RUN pip3 install llvmlite==0.35

# install deepspeech.pytorch
ADD . /workspace/Robust_ASR
RUN cd Robust_ASR; pip3 install -r requirements.txt

#setting up workspace
WORKDIR /workspace/Robust_ASR
