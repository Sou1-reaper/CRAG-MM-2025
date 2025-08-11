FROM python:3.10-slim-bookworm

RUN pip install --progress-bar off --no-cache-dir -U pip==21.0.1
RUN pip install --progress-bar off --no-cache-dir vllm==0.7.3  # We need at least 0.6.2 to support LLaMA3.2-Vision
COPY requirements.txt /tmp/requirements.txt
RUN pip install --progress-bar off --no-cache-dir -r /tmp/requirements.txt

WORKDIR /home/aicrowd
COPY . .
