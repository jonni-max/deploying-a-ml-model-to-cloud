#!/bin/bash

# Build steps on the render web server. 
# This will be called on every push to github.

pip install -r requirements.txt &&
aws_cred_export.sh &&
dvc pull -r jps3

