#! /bin/bash

# This script builds a docker image,
# tags it and pushes it to the Heroku registry.
# The script is optimized for OS X with M1.
# To use ot with Linux/Windows, check:
# https://github.com/mxagar/mlops_udacity/blob/main/03_Deployment/MLOpsND_Deployment.md#73-heroku-docker-deployment

# The app must have been created.
# To do it via the web interface:
# Heroku Dashboard 
#   New: Create new app > App name: <app-name>, e.g., census-salary-container
#   Deployment method: Container registry
# To do it via CLI:
#   heroku create <app-name>
#   heroku create census-salary-container

##
## VARIABLES: change them depending on the application
##

APP_NAME=census-salary-container
IMAGE_NAME=census_model_api
PROCESS_TYPE=web
# WARNING: Make sure to run `heroku login` before executing this script

##
## COMMANDS
##

# 0) LOG IN, if not done that yet
# heroku login
echo "Bulding and pushing app image to Heroku."
echo "This will work only if you already run"
echo "heroku login"

# 1) Sign in to Heroku Container Registry: registry.heroku.com
echo "Signing in to Heroku registry..."
heroku container:login

# 2) Build image
# Mac M1: Build for platform compatible with Heroku
# https://stackoverflow.com/questions/66982720/keep-running-into-the-same-deployment-error-exec-format-error-when-pushing-nod
# https://alex-shim.medium.com/building-a-docker-image-for-aws-x86-64-ec2-instance-with-mac-m1-chip-44a3353516b9
echo "Building image..."
docker buildx build --platform linux/amd64 -t ${IMAGE_NAME}:latest .

# 3) Tag the local image with the registry.
# Local image: ${IMAGE_NAME}:latest
# Registry image:
#   registry.heroku.com//${APP_NAME}/${PROCESS_TYPE}
#   registry.heroku.com/census-salary-container/web
echo "Tagging image..."
docker tag ${IMAGE_NAME}:latest registry.heroku.com/${APP_NAME}/${PROCESS_TYPE}

# 4) Push image to the Heroku registry
echo "Pushing image..."
docker push registry.heroku.com/${APP_NAME}/${PROCESS_TYPE}

# 5) Deploy: Release the newly pushed images to deploy your app.
echo "Releasing app..."
heroku container:release web --app ${APP_NAME}

# 6) Open the app in your browser
echo "Opening app on browser..."
heroku open --app ${APP_NAME}
# https://${APP_NAME}.herokuapp.com
# https://census-salary-container.herokuapp.com