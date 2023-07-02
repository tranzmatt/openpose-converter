#!/usr/bin/env python3
import cv2
import os
import sys
import fnmatch
import rembg

# set the default image pattern
image_pattern = "*.768.jpg"

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

                # read the input image
                image = cv2.imread(image_file)

                # isolate the foreground using rembg
                with rembg.open(image) as f:
                    image_alpha = f.read()

                # convert to grayscale
                gray = cv2.cvtColor(image_alpha, cv2.COLOR_RGBA2GRAY)

                # set the low and high thresholds
                low_threshold = 50
                high_threshold = 150

                # perform canny edge detection
                canny_output = cv2.Canny(gray, low_threshold, high_threshold)

                # invert the colors of the canny output
                inverted_output = 255 - canny_output

                # apply binary thresholding using Otsu's method
                _, threshold_output = cv2.threshold(canny_output, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

                # construct the output filenames with the appropriate suffixes
                filename, ext = os.path.splitext(image_file)
                canny_output_filename = f"{filename}.rembg.canny_{low_threshold}_{high_threshold}.png"
                threshold_output_filename = f"{filename}.rembg.threshold.png"
                inverted_output_filename = f"{filename}.rembg.threshold_inverted.png"
                rembg_output_filename = f"{filename}.rembg.png"

                # save the output images
                cv2.imwrite(canny_output_filename, canny_output)
                cv2.imwrite(threshold_output_filename, threshold_output)
                cv2.imwrite(inverted_output_filename, inverted_output)
                cv2.imwrite(rembg_output_filename, image_alpha)
        elif entry.is_dir():
            process_files(entry.path)  # recurse into subdirectories

# start processing files in the current directory and its subdirectories
process_files('.')

