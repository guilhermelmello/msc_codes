# HOW TO BUILD:
# docker build \
#   --build-arg USER_ID=$(id -u ${USER}) \
#   --build-arg GROUP_ID=$(id -g ${USER}) \
#   --build-arg USER_NAME=$(id -nu ${USER}) \
#   --build-arg GROUP_NAME=$(id -ng ${USER}) \
#   -t gmello_mlm:1.0 .
#
# HOW TO RUN:
# docker run -it --rm \
#     --user "$(id -u):$(id -g)" \
#     gmello_mlm:1.0

FROM nvcr.io/nvidia/pytorch:23.04-py3

ARG USER_ID
ARG GROUP_ID
ARG GROUP_NAME
ARG USER_NAME

RUN groupadd -g ${GROUP_ID} ${GROUP_NAME}
RUN useradd -l -u ${USER_ID} -g ${GROUP_ID} ${USER_NAME}
RUN install -d -m 0755 -o ${USER_NAME} -g ${GROUP_NAME} /home/${USER_NAME}
USER ${USER_NAME}


COPY requirements.txt /tmp/requirements.txt
ENV HF_DATASETS_CACHE /workspace/hf_cache
RUN pip install --upgrade pip
RUN pip install -r /tmp/requirements.txt

# pip is using this folder
ENV PATH /home/gmello/.local/bin:$PATH
