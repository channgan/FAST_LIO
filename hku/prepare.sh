#!/usr/bin/env bash
# Materialize the pinned algorithm in the build tree. Only its ROS I/O is adapted.
set -euo pipefail
source_dir=$(cd "$(dirname "$0")/.." && pwd)
dest=$1
commit=cc3c9e8b1469b33b8072764d6db2348ed6dc597f
mkdir -p "$dest"
git -C "$source_dir" archive "$commit" include src | tar -x -C "$dest"
tree=$(git -C "$source_dir" rev-parse "$commit:include/ikd-Tree")
git -C "$source_dir/include/ikd-Tree" archive "$tree" | tar -x -C "$dest/include/ikd-Tree"
patch --quiet --batch -d "$dest" -p1 < "$source_dir/hku/output.patch"
