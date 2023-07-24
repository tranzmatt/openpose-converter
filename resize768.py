#!/usr/bin/env python3

import os
import sys
import cv2

# Function to resize an image and save it with the specified suffix
def resize_image(file, suffix, output_folder):
    # Extract the base directory, filename, and extension
    directory = os.path.dirname(file)
    filename = os.path.basename(file)
    base, extension = os.path.splitext(filename)

    # Check if the resized image already exists
    resized_image = os.path.join(directory, output_folder, f"{base}.{suffix}{extension}")
    if os.path.isfile(resized_image):
        print(f"Skipping {file} - Resized image already exists.")
        return

    # Check if the current directory contains the output folder
    if output_folder in directory:
        print(f"Skipping {file} - Already in {output_folder} subfolder.")
        return

    # Create the output folder if it doesn't exist
    os.makedirs(os.path.join(directory, output_folder), exist_ok=True)

    # Load the image
    image = cv2.imread(file)

    # Resize the image while maintaining aspect ratio
    max_size = 768

    # Determine the aspect ratio
    original_height, original_width = image.shape[:2]
    if original_width > original_height: # landscape mode
        aspect_ratio = original_width / original_height
        new_width = max_size
        new_height = int(new_width / aspect_ratio)
    else: # portrait mode or square
        aspect_ratio = original_height / original_width
        new_height = max_size
        new_width = int(new_height / aspect_ratio)

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Save the resized image
    output_file_path = os.path.join(directory, output_folder, f"{base}.{suffix}{extension}")
    print(f"Saving to {output_file_path}")
    cv2.imwrite(output_file_path, resized_image)

# Function to recursively search for image files in subfolders
def search_and_resize(folder, output_folder):
    # Find all image files in the current folder and its subfolders
    image_extensions = (".jpg", ".jpeg", ".png", ".gif")
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(image_extensions):
                # Resize each image
                resize_image(os.path.join(root, file), "768", output_folder)

# Check if folder path and output folder name are provided as command-line arguments
if len(sys.argv) != 3:
    print("Please provide both a folder path and an output folder name as arguments.")
    sys.exit(1)

# Start searching for image files and resizing them
search_and_resize(sys.argv[1], sys.argv[2])

