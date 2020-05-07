FROM openvino/ubuntu18_runtime

WORKDIR /app

COPY requirements.txt .

COPY . /app

