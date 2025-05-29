# FROM nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . /app

# Set the working directory
WORKDIR /app

# copy the script to the working directory and set permissions
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# command to run the script to source the environment variables
ENTRYPOINT ["/entrypoint.sh"]
# Command to run the Python script
CMD ["python3", "iris/execute.py"]