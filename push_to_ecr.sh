#! /bin/bash

# This script builds a docker image,
# tags it and pushes it to AWS ECR.
# The script is optimized for OS X with M1,
# but alternatives for Linux/Windows are provided in the comments.

##
## VARIABLES: change them depending on the application
##

APP_NAME=census-model-api
REPOSITORY_NAME=census-model-api
AWS_ACCOUNT_ID=077073585279
AWS_DEFAULT_REGION=us-east-1
# WARNING: Make sure to run `aws configure` before executing this script
# With it, we set the AWS access key and the key secret/password

##
## COMMANDS
##

# 0) CONFIGURE, if not done that yet
# aws configure
echo "Bulding and pushing app image to AWS ECR."
echo "This will work only if you already run"
echo "aws configure"

# 1) BUILD the image
# Mac M1: Build for platform compatible with Heroku
#   docker buildx build --platform linux/amd64 -t <app-name>:latest .
# Windows / Linux:
#   docker build -t <app-name>:latest .
# References:
#   https://stackoverflow.com/questions/66982720/keep-running-into-the-same-deployment-error-exec-format-error-when-pushing-nod
#   https://alex-shim.medium.com/building-a-docker-image-for-aws-x86-64-ec2-instance-with-mac-m1-chip-44a3353516b9
# NOTE: to make things easier, I chose <app-name> = <repository-name> = census-model-api
echo "Building image..."
docker buildx build --platform linux/amd64 -t ${APP_NAME}:latest .

# 2) TAG the image with the AWS ECR repository
# We nee the AWS_ACCOUNT_ID, which is in the URI of the repository we created
# We can either copy and use that ID or set it as an environment variable
#   docker tag <app-name>:latest ${AWS_ACCOUNT_ID}.dkr.ecr.<region>.amazonaws.com/<repository-name>:latest
echo "Tagging image..."
docker tag ${APP_NAME}:latest ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_DEFAULT_REGION}.amazonaws.com/${REPOSITORY_NAME}:latest

# 3) Log in to the ECR
# We need a password which can be obtained with this command
#   aws ecr get-login-password --region <region>
# HOWEVER, it is more practicle to pipe that password to the login command
# Reference: https://awscli.amazonaws.com/v2/documentation/api/2.0.34/reference/ecr/get-login-password.html
echo "Logging to AWS ECR..."
aws ecr get-login-password --region ${AWS_DEFAULT_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_DEFAULT_REGION}.amazonaws.com
# We should get: "Login Succeeded"

# 4) PUSH image to ECR
echo "Pushing image..."
# We push the tagged image:
#   docker push ${AWS_ACCOUNT_ID}.dkr.ecr.<region>.amazonaws.com/<repository-name>:latest
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_DEFAULT_REGION}.amazonaws.com/${REPOSITORY_NAME}:latest
# We wait for it to be uploaded
# Then, we can check that the image is in the repository:
# AWE ECR > Repositories: census-model-api > Images