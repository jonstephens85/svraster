#!/bin/bash
set -e

# Usage: ./run_colmap.sh <project-name> [--matcher exhaustive]

# === Parse Args ===
PROJECT_NAME=$1
MATCHER="sequential"  # default matcher

# Check for matcher override
if [ "$2" == "--matcher" ] && [ -n "$3" ]; then
  MATCHER="$3"
fi

if [ -z "$PROJECT_NAME" ]; then
  echo "Usage: ./run_colmap.sh <project-name> [--matcher exhaustive|sequential]"
  exit 1
fi

# === Paths ===
PROJECT_DIR="data/${PROJECT_NAME}"
DB_PATH="${PROJECT_DIR}/database.db"
IMAGE_PATH="${PROJECT_DIR}/images"
OUTPUT_PATH="${PROJECT_DIR}/colmap/sparse"

# === Check COLMAP ===
if ! command -v colmap &> /dev/null; then
  echo "❌ COLMAP not found in PATH."
  if command -v conda &> /dev/null; then
    echo "➡️  Attempting to install COLMAP via Conda..."
    conda install -y -c conda-forge colmap || {
      echo "❌ Conda install failed. Please install COLMAP manually."
      exit 1
    }
  else
    echo "❌ Conda not available. Please install COLMAP manually."
    exit 1
  fi
fi

# === Feature Extraction ===
colmap feature_extractor \
    --database_path "$DB_PATH" \
    --image_path "$IMAGE_PATH" \
    --ImageReader.camera_model PINHOLE \
    --ImageReader.single_camera 1

# === Matching ===
if [ "$MATCHER" == "exhaustive" ]; then
  echo "🔁 Running exhaustive matcher..."
  colmap exhaustive_matcher --database_path "$DB_PATH"
else
  echo "🔁 Running sequential matcher (default)..."
  colmap sequential_matcher --database_path "$DB_PATH"
fi

mkdir -p "$OUTPUT_PATH"

# === Mapping ===
colmap mapper \
    --database_path "$DB_PATH" \
    --image_path "$IMAGE_PATH" \
    --output_path "$OUTPUT_PATH"