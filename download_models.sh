#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p models
curl -fL --retry 3 "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/colorization_deploy_v2.prototxt" -o models/colorization_deploy_v2.prototxt
curl -fL --retry 3 "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/pts_in_hull.npy" -o models/pts_in_hull.npy
curl -fL --retry 3 "https://github.com/richzhang/colorization/releases/download/v1.0/colorization_release_v2.caffemodel" -o models/colorization_release_v2.caffemodel
test "$(stat -f%z models/colorization_release_v2.caffemodel)" -gt 1000000 || { echo "Model download incomplete"; exit 1; }
echo "Colorization model ready."
