#!/bin/bash

set -e

# Check if COLMAP is installed
if ! command -v colmap &> /dev/null; then
  echo "❌ COLMAP is not installed or not in your PATH."
  echo ""
  echo "Install options:"
  echo "  • Conda: conda install -c conda-forge colmap"
  echo "  • Ubuntu (via PPA): sudo add-apt-repository ppa:colmap/colmap && sudo apt update && sudo apt install colmap"
  echo "  • From source: https://colmap.github.io/install.html"
  exit 1
fi

# Check for project name argument
if [ -z "$1" ]; then
  echo "Usage: ./run_colmap.sh <project-name>"
  exit 1
fi

PROJECT_NAME=$1
PROJECT_DIR="data/${PROJECT_NAME}"
DB_PATH="${PROJECT_DIR}/database.db"
IMAGE_PATH="${PROJECT_DIR}/images"
OUTPUT_PATH="${PROJECT_DIR}/colmap/sparse"

# Step 1: Feature Extraction
colmap feature_extractor \
    --database_path "$DB_PATH" \
    --image_path "$IMAGE_PATH" \
    --ImageReader.camera_model PINHOLE \
    --ImageReader.single_camera 1

# Step 2: Exhaustive Matching
colmap exhaustive_matcher \
    --database_path "$DB_PATH"

# Step 3: Mapping
colmap mapper \
    --database_path "$DB_PATH" \
    --image_path "$IMAGE_PATH" \
    --output_path "$OUTPUT_PATH"
