#!/bin/bash
set -e

# Data download script for CCPP and other datasets

echo "Checking if data exists..."
if [ -d "raw" ]; then
    echo "Data directory 'raw' already exists."
else
    mkdir -p raw
    echo "Created 'raw' directory."
    echo "Please download the CCPP dataset manually and extract to data/raw/ccpp/"
    echo "Instructions provided in README_data.md"
fi
