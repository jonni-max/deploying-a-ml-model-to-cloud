#!/bin/bash

# Build steps on the render web server. 
# This will be called on every push to github.

pip install -r requirements.txt &&
source ./aws_cred_export &&
dvc pull -r jps3

