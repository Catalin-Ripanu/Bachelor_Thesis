# Use an official Miniconda image from Docker Hub
FROM continuumio/miniconda3

# Copy the environment.yml file into the Docker image
COPY environment.yml /tmp/environment.yml

# Create the conda environment from the environment.yml file
RUN conda env create -f /tmp/environment.yml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Set the default environment for subsequent commands
ENV PATH /opt/conda/envs/myenv/bin:$PATH

# Install any additional dependencies
RUN conda install -n myenv pip

# Optional: Set an entrypoint or command to run on container start
ENTRYPOINT ["conda", "run", "-n", "myenv", "python"]

# Optionally, copy your application code into the image
# COPY . /app
# WORKDIR /app

# Command to run your application (e.g., a Python script)
# CMD ["your_script.py"]
