"""
Download script for the WMT 2014 dataset.
"""

import os
import sys

def main():
    data_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(data_dir, "raw")
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)
        
    print("Instructions to download WMT 2014 dataset...")
    print("Downloading dataset...")
    # Add actual download logic here depending on source.
    print("Done.")

if __name__ == "__main__":
    main()
