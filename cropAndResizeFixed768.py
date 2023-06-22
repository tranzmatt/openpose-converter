#!/usr/bin/env python3

import numpy as np
from PIL import Image, ImageDraw
import glob
import os
import argparse
import io

from pathlib import Path
from rembg import remove, new_session

model_name = 'u2net_human_seg'
#session = new_session(model_name)
session = new_session()

def crop_and_resize(image_path, output_dir):
    # Construct new file names
    base_name, ext = os.path.splitext(os.path.basename(image_path))
    new_file_name = f"{base_name}.cr768.png"  # The output will be in PNG format to maintain transparency
    foreground_file_name = f"{base_name}.foreground.png"  # Foreground image file name
    bounding_box_file_name = f"{base_name}.bb.png"  # Bounding box image file name
    cropped_file_name = f"{base_name}.cropped.png"  # Cropped image file name

    output_path = os.path.join(output_dir, new_file_name)
    foreground_output_path = os.path.join(output_dir, foreground_file_name)  # Path for foreground image
    bounding_box_output_path = os.path.join(output_dir, bounding_box_file_name)  # Path for bounding box image
    cropped_output_path = os.path.join(output_dir, cropped_file_name)  # Path for cropped image

    padding_size = 200

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
        img_no_bg_bytes = remove(img, session=session)

        # Convert to PIL Image
        img_no_bg = Image.open(io.BytesIO(img_no_bg_bytes))

        # Save the foreground image
        img_no_bg.save(foreground_output_path)

    # Get the bounding box of the foreground object
    non_transparent = np.array(img_no_bg)[:, :, 3] > 0
    rows = np.any(non_transparent, axis=1)
    cols = np.any(non_transparent, axis=0)
    min_row, max_row = np.where(rows)[0][[0, -1]]
    min_col, max_col = np.where(cols)[0][[0, -1]]

    # Calculate the original width and height
    width = max_col - min_col
    height = max_row - min_row
    #print(f"Prepad dim = {width} x {height}")

    # Determine the longer and shorter dimensions
    if width > height:
        min_col = max(0, min_col - padding_size)
        max_col = min(img_no_bg.width - 1, max_col + padding_size)
        width = max_col - min_col
        longer_dim = width
        shorter_dim = height
    else:
        min_row = max(0, min_row - padding_size)
        max_row = min(img_no_bg.height - 1, max_row + padding_size)
        height = max_row - min_row
        longer_dim = height
        shorter_dim = width

    #print(f"Post-pad dim = {width} x {height}")

    # Calculate the desired aspect ratio based on the shorter dimension
    desired_aspect_ratio = 2 / 3 if shorter_dim == width else 3 / 2

    # Adjust the bounding box to meet the desired aspect ratio
    increase = (longer_dim * desired_aspect_ratio) - shorter_dim
    if shorter_dim == width:
        # Increase the shorter dimension until the desired aspect ratio is met
        min_col = max(0, min_col - increase // 2)
        max_col = min(img_no_bg.width - 1, max_col + increase // 2)
    else:
        # Increase the shorter dimension until the desired aspect ratio is met
        min_row = max(0, min_row - increase // 2)
        max_row = min(img_no_bg.height - 1, max_row + increase // 2)

    # Calculate the adjusted bounding box
    new_bbox = (min_col, min_row, max_col, max_row)

    # Save the bounding box image
    #bounding_box_img = img_no_bg.copy().convert("RGB")
    #draw = ImageDraw.Draw(bounding_box_img)
    #draw.rectangle(new_bbox, outline="red", width=3)
    #bounding_box_img.save(bounding_box_output_path)

    # Crop the original image using the adjusted bounding box
    img = Image.open(image_path)
    cropped_img = img.crop(new_bbox)

    # Save the cropped image
    cropped_img.save(cropped_output_path)

    # Get the dimensions of the cropped image
    cropped_width, cropped_height = cropped_img.size

    # Resize the cropped image while maintaining the aspect ratio
    if cropped_width > cropped_height:
        new_width = 768
        new_height = int(cropped_height * new_width / cropped_width)
    else:
        new_height = 768
        new_width = int(cropped_width * new_height / cropped_height)

    resized_img = cropped_img.resize((new_width, new_height), Image.LANCZOS)
    resized_img.save(output_path)


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

