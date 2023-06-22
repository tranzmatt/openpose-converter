#!/usr/bin/env python3

from rembg import remove
import numpy as np
from PIL import Image
import glob
import os
import argparse
import io

def crop_and_resize(image_path, output_dir, size=768, padding=100):
    # Construct new file names
    base_name, ext = os.path.splitext(os.path.basename(image_path))
    new_file_name = f"{base_name}.cr{size}.png" # The output will be in PNG format to maintain transparency
    foreground_file_name = f"{base_name}_foreground.png"  # Foreground image file name

    output_path = os.path.join(output_dir, new_file_name)
    foreground_output_path = os.path.join(output_dir, foreground_file_name)  # Path for foreground image

    # If output file already exists, ignore this image
    if os.path.exists(output_path):
        return

    # If foreground file already exists, load it, else create it
    if os.path.exists(foreground_output_path):
        img_no_bg = Image.open(foreground_output_path)
    else:
        # Load image
        with open(image_path, 'rb') as img_file:
            img = img_file.read()

        # Remove background
        img_no_bg_bytes = remove(img)

        # Convert to PIL Image
        img_no_bg = Image.open(io.BytesIO(img_no_bg_bytes))

        # Save the foreground image
        img_no_bg.save(foreground_output_path)

    # Find non-transparent pixels
    non_transparent = np.array(img_no_bg)[:, :, :3].any(-1)

    # Get the bounding box
    non_zero_columns = np.where(non_transparent.max(axis=0))[0]
    non_zero_rows = np.where(non_transparent.max(axis=1))[0]

    # Check if there are any non-transparent pixels
    if non_zero_rows.size == 0 or non_zero_columns.size == 0:
        print(f"No foreground in image {image_path}. Skipping...")
        return

    # Create the bounding box coordinates with padding, ensuring it doesn't go out of image boundaries
    left = max(min(non_zero_columns) - padding, 0)
    upper = max(min(non_zero_rows) - padding, 0)
    right = min(max(non_zero_columns) + padding, img_no_bg.width - 1)
    lower = min(max(non_zero_rows) + padding, img_no_bg.height - 1)
    cropBox = (left, upper, right, lower)

    # Open the original image for cropping
    img_original = Image.open(image_path)

    # Crop the original image to this bounding box
    cropped = img_original.crop(cropBox)

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
        print(f"Processing {file}")
        crop_and_resize(file, output_dir)
    for file in glob.iglob(os.path.join(args.input_dir, "*.[pP][nN][gG]")):
        print(f"Processing {file}")
        crop_and_resize(file, output_dir)
