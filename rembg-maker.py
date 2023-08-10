#!/usr/bin/env python3

import cv2
import os
import fnmatch
import argparse
from rembg import remove, new_session

session = new_session()

DETECTOR_NAMES = ["canny", "hed", "mlsd", "midas", "openpose", "uniformer"]


# get the images files
def get_image_files(input_path, filters=None):
    valid_extensions = [".jpg", ".JPG", ".png", ".PNG"]
    if os.path.isdir(input_path):
        image_files = []
        for ext in valid_extensions:
            image_files.extend(glob.glob(os.path.join(input_path, '**', '*' + ext), recursive=True))

        # Filter out images with detector names
        image_files = [img for img in image_files if not any(detector in img for detector in DETECTOR_NAMES)]

        # Apply additional filters if provided
        if filters:
            filtered_files = []
            for pattern in filters:
                filtered_files.extend(fnmatch.filter(image_files, pattern))
            image_files = filtered_files

    elif os.path.isfile(input_path) and any(input_path.endswith(ext) for ext in valid_extensions):
        image_files = [input_path]
    else:
        image_files = []

    return image_files


# recursive function to process files in a directory
def process_files(directory, image_patterns):
    for entry in os.scandir(directory):
        if entry.is_file():
            file_lower = entry.name.lower()
            if any(fnmatch.fnmatchcase(file_lower, pattern) for pattern in image_patterns) \
                    and "rembg" not in file_lower:
                image_file = entry.path
                print(f"Processing {image_file}")
                filename, ext = os.path.splitext(image_file)

                # read the input image
                input_image = cv2.imread(image_file)

                # isolate the foreground using rembg
                rembg_image = remove(input_image, session=session, bgcolor=(255, 255, 255, 1))
                rembg_image_filename = f"{filename}.rembg.png"
                cv2.imwrite(rembg_image_filename, rembg_image)

        elif entry.is_dir():
            process_files(entry.path, image_patterns)  # recurse into subdirectories


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images in a specified directory.")
    parser.add_argument('-p', '--path', default='.', help="Path to the directory to process.")
    parser.add_argument('-f', '--filter', nargs='+', default=["*.jpg", "*.png"],
                        help="Image filters (e.g. *.jpg *.png).")

    args = parser.parse_args()

    # start processing files in the specified directory and its subdirectories
    process_files(args.path, args.filter)
