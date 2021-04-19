#!/bin/bash
# This script assumes you have cloned the ray environment, it has been activated,
# and you're running this script from *here*
cd ..
TMPDIR=$LOCAL_SCRATCH pip install -r requirements.txt
TMPDIR=$LOCAL_SCRATCH pip install -e .
conda install -c conda-forge libjpeg-turbo -y
conda install pillow rtree tensorflow shapely -y
easy_install /projects/cavs/Carla/carla-0.9.5/PythonAPI/carla/dist/carla-0.9.5-py3.5-linux-x86_64.egg
cd hpc
