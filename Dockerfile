FROM python:3.12

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the entire project
COPY . /app

# Set the working directory
WORKDIR /app

# Command to run the Python script
CMD ["python3", "iris/execute.py"]