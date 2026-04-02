# WINDOWS

## SETUP

Open Docker Desktop

Run these in the project root:

docker pull tviskaron/mats-lp
docker tag tviskaron/mats-lp mats-lp

## RUN

docker run --rm -ti -v ${PWD}:/code -w /code mats-lp python3 main.py

After the program finishes you a simulation of it should be created in /renders