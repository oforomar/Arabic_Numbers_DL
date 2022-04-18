FROM tensorflow/tensorflow:2.8.0-gpu-jupyter

RUN mkdir -p /project/dataset

RUN apt update

RUN apt install ffmpeg libsm6 libxext6  -y

RUN pip install cookiecutter pandas sklearn opencv-contrib-python

WORKDIR /project

COPY ./dataset2.zip ./project/dataset

EXPOSE 8888
