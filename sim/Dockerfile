FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y \
        g++ \
        libx11-6 \
        libgl1 \
        libglu1 \
        libxrender1 \
        libxext6 \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install pybullet

WORKDIR /workspace

CMD ["python", "simulate_urdf.py"]

