#!/usr/bin/env python3

import cv2
import numpy as np
from PIL import Image
import glob
import os
import argparse

def crop_and_resize(image_path, output_dir, size=768):
    # Construct new file name
    base_name, ext = os.path.splitext(os.path.basename(image_path))
    new_file_name = f"{base_name}.cr{size}{ext.lower()}"

    output_path = os.path.join(output_dir, new_file_name)

    # If output file already exists, ignore this image
    if os.path.exists(output_path):
        return

    # Load image
    img = cv2.imread(image_path)

    # Set up the initial bounding rectangle
    rect = (50, 50, 450, 290)

    # Create a mask similar to the input image
    mask = np.zeros(img.shape[:2], np.uint8)

    # Create 2 arrays for the GrabCut algorithm
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Apply GrabCut algorithm with 5 iterations
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # Apply mask to isolate object
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Multiply per-element mask with the original image so that non-ROI is discarded
    img = img * mask2[:, :, np.newaxis]

    # Find contour for the non-zero regions of the mask
    contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming there's only one contour of interest, get its bounding rectangle
    x, y, w, h = cv2.boundingRect(contours[0])

    # Crop the image to the bounding rectangle
    cropped = img[y: y + h, x: x + w]

    # Convert image from OpenCV to PIL format for easier resizing
    cropped_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))

    # Get the larger dimension
    max_dim = max(cropped_pil.size)

    # Compute new dimensions keeping aspect ratio
    new_dims = (size, int(cropped_pil.size[1] * size / max_dim)) if cropped_pil.size[0] > cropped_pil.size[1] else (int(cropped_pil.size[0] * size / max_dim), size)

    # Resize image
    resized = cropped_pil.resize(new_dims, Image.ANTIALIAS)

    # Save image
    resized.save(output_path)

if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Crop and resize images.')
    parser.add_argument('input_dir', type=str, help='Input directory for images')
    parser.add_argument('output_subdir', type=str, help='Output subdirectory for processed images')

    args = parser.parse_args()

    # Make sure output directory exists
    output_dir = os.path.join(args.input_dir, args.output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    # Get all .jpg and .png files in the input directory (case insensitive)
    for file in glob.iglob(os.path.join(args.input_dir, "*.[jJ][pP][gG]")):
        crop_and_resize(file, output_dir)
    for file in glob.iglob(os.path.join(args.input_dir, "*.[pP][nN][gG]")):
        crop_and_resize(file, output_dir)

