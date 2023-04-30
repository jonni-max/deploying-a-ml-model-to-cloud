#!/bin/bash

# Deployment on the render web server. This will be called on every push to github.

pip install -r requirements.txt &&
dvc pull jps3

