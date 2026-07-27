FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

# System deps
RUN apt-get update && apt-get install -y git wget build-essential ninja-build && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Project
WORKDIR /workspace
COPY . .
RUN pip install -e .

# Default: show help
CMD ["python", "train.py", "--help"]
