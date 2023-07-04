#!/usr/bin/env python3
import cv2
import os
import sys
import fnmatch
from rembg import remove, new_session
import numpy as np

# set the default image pattern
image_pattern = "*.768.jpg"

session = new_session()

# check if a command-line argument is provided
if len(sys.argv) > 1:
    image_pattern = sys.argv[1]

# convert the pattern to lowercase
image_pattern_lower = image_pattern.lower()

# recursive function to process files in a directory
def process_files(directory):
    for entry in os.scandir(directory):
        if entry.is_file():
            file_lower = entry.name.lower()
            if fnmatch.fnmatchcase(file_lower, image_pattern_lower):
                image_file = entry.path
                print(f"Processing {image_file}")
                filename, ext = os.path.splitext(image_file)

                # read the input image
                input_image = cv2.imread(image_file)

                # isolate the foreground using rembg
                #rembg_image_tmp = remove(input_image, session=session, bgcolor=(255,255,255,1))
                #rembg_image = remove(rembg_image_tmp, session=session)
                rembg_image = remove(input_image, session=session, bgcolor=(255,255,255,1))
                rembg_image_filename = f"{filename}.rembg.png"
                cv2.imwrite(rembg_image_filename, rembg_image)

                # set the low and high thresholds
                low_threshold = 100
                high_threshold = 200

                # perform canny edge detection
                #canny_image = cv2.Canny(gray_image, low_threshold, high_threshold)
                canny_image = cv2.Canny(rembg_image, low_threshold, high_threshold)
                canny_image_filename = f"{filename}.rembg.canny_{low_threshold}_{high_threshold}.png"
                cv2.imwrite(canny_image_filename, canny_image)

                # invert the colors of the canny output
                inverted_image = 255 - canny_image

                # apply binary thresholding using Otsu's method
                _, threshold_image = cv2.threshold(canny_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

                # construct the output filenames with the appropriate suffixes
                #threshold_image_filename = f"{filename}.rembg.threshold.png"
                #inverted_image_filename = f"{filename}.rembg.threshold_inverted.png"

                # save the output images
                #cv2.imwrite(threshold_image_filename, threshold_image)
                #cv2.imwrite(inverted_image_filename, inverted_image)
        elif entry.is_dir():
            process_files(entry.path)  # recurse into subdirectories

# start processing files in the current directory and its subdirectories
process_files('.')

