#!/usr/bin/env python3
import cv2
import os
import sys
import fnmatch
from rembg import remove, new_session
import numpy as np

# set the default image patterns
image_patterns = ["*.jpg", "*.png"]

session = new_session()

# check if a command-line argument is provided
if len(sys.argv) > 1:
    image_patterns = [sys.argv[1]]

# recursive function to process files in a directory
def process_files(directory):
    for entry in os.scandir(directory):
        if entry.is_file():
            file_lower = entry.name.lower()
            if any(fnmatch.fnmatchcase(file_lower, pattern) for pattern in image_patterns) and "rembg" not in file_lower:
                image_file = entry.path
                print(f"Processing {image_file}")
                filename, ext = os.path.splitext(image_file)

                # read the input image
                input_image = cv2.imread(image_file)

                # isolate the foreground using rembg
                rembg_image = remove(input_image, session=session, bgcolor=(255,255,255,1))
                rembg_image_filename = f"{filename}.rembg.png"
                cv2.imwrite(rembg_image_filename, rembg_image)

        elif entry.is_dir():
            process_files(entry.path)  # recurse into subdirectories

# start processing files in the current directory and its subdirectories
process_files('.')

