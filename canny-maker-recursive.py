#!/usr/bin/env python3

import cv2
import glob
import os
import fnmatch

# specify the wildcard pattern for input images
image_pattern = "*.768.JPG"

# convert the pattern to lowercase
image_pattern_lower = image_pattern.lower()

# get a list of all files matching the wildcard pattern
image_files = []
for root, dirs, files in os.walk('.'):
    for file in files:
        # convert the filename to lowercase
        file_lower = file.lower()
        if fnmatch.fnmatchcase(file_lower, image_pattern_lower):
            image_files.append(os.path.join(root, file))

# iterate over the image files
for image_file in image_files:
    # read the input image
    image = cv2.imread(image_file)

    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
    canny_output_filename = f"{filename}.canny_{low_threshold}_{high_threshold}.png"
    threshold_output_filename = f"{filename}.threshold.png"
    inverted_output_filename = f"{filename}.threshold_inverted.png"

    # save the output images
    cv2.imwrite(canny_output_filename, canny_output)
    cv2.imwrite(threshold_output_filename, threshold_output)
    cv2.imwrite(inverted_output_filename, inverted_output)

