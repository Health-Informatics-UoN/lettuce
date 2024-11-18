# Use the official Python 3.10 Alpine base image
FROM ubuntu:22.04

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    python3-venv \
    build-essential \
    cmake \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a symlink for python to point to python3.11
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Upgrade pip and set python3.11 as the default python3
RUN python3.11 -m pip install --upgrade pip && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Set up non-root user
RUN useradd -m appuser

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code to the container
COPY . .

# Expose the port on which the application will run
EXPOSE 8000

# Set permissions for the app directory
RUN chown -R appuser:appuser /app

# Specify the user to run the application
USER appuser

# Set the entrypoint
ENTRYPOINT ["uvicorn"]

# Run the application
CMD ["app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]