FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# System dependencies
RUN apt-get update && apt-get install -y \
    git wget curl build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install project
COPY . .
RUN pip install -e . --no-deps

# Expose port for Jupyter
EXPOSE 8888

# Default: show help
CMD ["python", "train.py", "--help"]
