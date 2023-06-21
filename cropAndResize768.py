#!/usr/bin/env python3

from rembg import remove
import numpy as np
from PIL import Image
import glob
import os
import argparse
import io

def crop_and_resize(image_path, output_dir, size=768):
    # Construct new file name
    base_name, ext = os.path.splitext(os.path.basename(image_path))
    new_file_name = f"{base_name}.cr{size}.png" # The output will be in PNG format to maintain transparency

    output_path = os.path.join(output_dir, new_file_name)

    # If output file already exists, ignore this image
    if os.path.exists(output_path):
        return

    # Load image
    with open(image_path, 'rb') as img_file:
        img = img_file.read()

    # Remove background
    img_no_bg_bytes = remove(img)

    # Convert to PIL Image
    img_no_bg = Image.open(io.BytesIO(img_no_bg_bytes))

    # Find non-transparent pixels
    non_transparent = np.array(img_no_bg)[:, :, :3].any(-1)

    # Get the bounding box
    non_zero_columns = np.where(non_transparent.max(axis=0))[0]
    non_zero_rows = np.where(non_transparent.max(axis=1))[0]
    cropBox = (min(non_zero_rows), max(non_zero_rows), min(non_zero_columns), max(non_zero_columns))

    # Crop the image to this bounding box
    cropped = img_no_bg.crop(cropBox)

    # Get the larger dimension
    max_dim = max(cropped.size)

    # Compute new dimensions keeping aspect ratio
    new_dims = (size, int(cropped.size[1] * size / max_dim)) if cropped.size[0] > cropped.size[1] else (int(cropped.size[0] * size / max_dim), size)

    # Resize image
    resized = cropped.resize(new_dims, Image.LANCZOS)

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

