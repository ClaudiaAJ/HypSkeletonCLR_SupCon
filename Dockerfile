# Use a CUDA-compatible base image with Python 3.10
FROM master.garching.cluster.campar.in.tum.de:10443/camp/ubuntu_22.04-python_3.10:latest

# Set non-interactive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and clean up
RUN apt update && apt install -y \
    wget \
    curl \
    git \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    python3-pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, and wheel
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Pre-install numpy
RUN python3 -m pip install --no-cache-dir numpy==1.24.4

# Install PyTorch and CUDA dependencies
RUN python3 -m pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --extra-index-url https://download.pytorch.org/whl/cu121

# Upgrade requests, and urllib3
RUN python3 -m pip install --upgrade requests urllib3

# Copy requirements.txt into the container
COPY requirements.txt /workspace/requirements.txt

# Install Python dependencies with ignore-installed to handle distutils conflicts
RUN python3 -m pip install --no-cache-dir --ignore-installed -r /workspace/requirements.txt

# Uninstall pre-installed PyTorch Geometric packages if present
RUN python3 -m pip uninstall -y \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    torch-geometric || true

# Install PyTorch Geometric dependencies
RUN python3 -m pip install --no-cache-dir torch-scatter -f https://data.pyg.org/whl/torch-2.1.2+cu121.html && \
    python3 -m pip install --no-cache-dir torch-sparse -f https://data.pyg.org/whl/torch-2.1.2+cu121.html && \
    python3 -m pip install --no-cache-dir torch-cluster -f https://data.pyg.org/whl/torch-2.1.2+cu121.html && \
    python3 -m pip install --no-cache-dir torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.2+cu121.html && \
    python3 -m pip install --no-cache-dir torch-geometric

# Copy application code into the container
COPY . /workspace

# Install the application in editable mode
RUN python3 -m pip install --no-cache-dir -e /workspace/torchlight

# Set the working directory
WORKDIR /workspace

# Default command to run the container
CMD ["bash"]
