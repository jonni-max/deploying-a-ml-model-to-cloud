#!/bin/bash

# Build steps on the render web server. 
# This will be called on every push to github.

pip install -r requirements_render.txt &&
source ./aws_cred_export &&
mkdir -p './tmp/dvc' &&
dvc config core.site_cache_dir './tmp/dvc' && 
dvc pull -r jps3 
#dvc config -vv --list

echo "DVC pulled data:"
dvc ls -R --dvc-only ./data
