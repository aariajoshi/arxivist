#!/usr/bin/env python3
"""
Data downloading script.
"""
import os
import sys

def main():
    print("Mamba training uses datasets like 'The Pile'.")
    print("These datasets are large and typically require external tools to download.")
    print("Please refer to data/README_data.md for instructions.")
    
    data_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(data_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    print(f"Created data directory at {raw_dir}")

if __name__ == "__main__":
    main()
