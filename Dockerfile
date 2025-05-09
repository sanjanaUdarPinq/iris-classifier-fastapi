FROM python:3.12

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the entire project
COPY . /app

# Set the working directory
WORKDIR /app

# copy the srcipt to the working directory and set permissions
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# command to run the script to source the environment variables
ENTRYPOINT ["/entrypoint.sh"]
# Command to run the Python script
CMD ["python3", "iris/execute.py"]