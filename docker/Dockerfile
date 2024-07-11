
FROM rostooling/setup-ros-docker:ubuntu-focal-ros-noetic-desktop-latest



RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*


RUN apt update && apt install ffmpeg libsm6 libxext6 -y

RUN apt-get update && apt-get install -y lsb-release && apt-get clean all
ARG DEBIAN_FRONTEND=noninteractive
RUN set -xe \
    && apt-get update \
    && apt-get install python3-pip
RUN pip install --upgrade pip
RUN pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
RUN set -xe \
    && apt-get update \
    && apt-get install python3 python3-pip -y

# Upgrade pip
RUN pip install --upgrade pip