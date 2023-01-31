# We can modify image/python version with
# docker build --build-arg IMAGE=python:3.8
# Otherwise, default: python:3.9.16
ARG IMAGE=python:3.9.16
FROM $IMAGE

# Create the user that will run the app
RUN adduser --disabled-password --gecos '' ml-api-user

# Create directory IN container and change to it
WORKDIR /opt/census_model

# Copy folder contents (unless the ones from .dockerignore) TO container
ADD . /opt/census_model/
# Install requirements
RUN pip install --upgrade pip
RUN pip install -r /opt/census_model/requirements.txt --no-cache-dir
RUN pip install .

# Change permissions
RUN chmod +x /opt/census_model/run.sh
RUN chown -R ml-api-user:ml-api-user ./

# Change user to the one created
USER ml-api-user

# Expose port
EXPOSE 8001

# Run web server, started by run.sh
CMD ["bash", "./run.sh"]


# Build the Dockerfile to create the image
# docker build -t <image_name[:version]> <path/to/Dockerfile>
#   docker build -t census_model_api:latest .
# 
# Check the image is there: watch the size (e.g., ~1GB)
#   docker image ls
#
# Run the container locally from a built image
# Recall to: forward ports (-p) and pass PORT env variable (-e)
# Optional: 
# -d to detach/get the shell back,
# --name if we want to choose conatiner name (else, one randomly chosen)
# --rm: automatically remove container after finishing (irrelevant in our case, but...)
#   docker run -d --rm -p 8001:8001 -e PORT=8001 --name census_model_app census_model_api:latest
#
# Check the API locally: open the browser
#   http://localhost:8001
#   Use the web API
# 
# Check the running containers: check the name/id of our container,
# e.g., census_model_app
#   docker container ls
#   docker ps
#
# Get a terminal into the container: in general, BAD practice
# docker exec -it <id|name> sh
#   docker exec -it census_model_app sh
#   (we get inside)
#   cd /opt/census_model
#   ls
#   cat logs/census_pipeline.log
#   exit
#
# Stop container and remove it (erase all files in it, etc.)
# docker stop <id/name>
# docker rm <id/name>
#   docker stop census_model_app
#   docker rm census_model_app
