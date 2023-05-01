#!/bin/bash

# Build steps on the render web server. 
# This will be called on every push to github.

pip install -r requirements_render.txt &&
source ./aws_cred_export &&
dvc pull -r jps3

echo "DVC pulled data:"
dvc ls -R --dvc-only ./data
