#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p models
BASE="https://storage.openvinotoolkit.org/repositories/datumaro/models/colorization"
for name in colorization_deploy_v2.prototxt pts_in_hull.npy colorization_release_v2.caffemodel; do
  if [ ! -s "models/$name" ]; then
    echo "Downloading $name..."
    curl --fail --location --retry 3 --connect-timeout 30 "$BASE/$name" -o "models/$name.part"
    mv "models/$name.part" "models/$name"
  fi
done
test "$(stat -f%z models/colorization_release_v2.caffemodel)" -gt 100000000 || { echo "Invalid model weights. Remove models/colorization_release_v2.caffemodel and retry."; exit 1; }
test "$(stat -f%z models/colorization_deploy_v2.prototxt)" -gt 1000 || { echo "Invalid prototxt"; exit 1; }
test "$(stat -f%z models/pts_in_hull.npy)" -gt 1000 || { echo "Invalid color centers"; exit 1; }
echo "Model files downloaded."
