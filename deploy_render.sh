#!/bin/bash

# Build steps on the render web server. 
# This will be called on every push to github.

pip install -r requirements_render.txt &&
source ./aws_cred_export &&
#dvc pull -r jps3
dvc ls-url  s3://udacity-mldevops-project-3

