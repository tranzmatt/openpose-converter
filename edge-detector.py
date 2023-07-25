#!/usr/bin/env python3

"""
Edge detection script using Kornia library.
"""
import argparse
import os
import glob
from multiprocessing import Pool
import torch

from PIL import Image
import torchvision.transforms as transforms

import cv2
try:
    import kornia as K
    from kornia.core import Tensor
except ImportError:
    raise ImportError("Please install the kornia library via pip: pip install kornia")


def edge_detection(filepath, detectors):
    """
    Function to perform edge detection on an image file with given detectors.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Add file existence check
    if not os.path.isfile(filepath):
        print(f"File {filepath} does not exist.")
        return

    img = K.io.load_image(filepath, K.io.ImageLoadType.RGB32).to(device)
    img = img[None]
    x_gray = K.color.rgb_to_grayscale(img)
    x_gray_pil = transforms.ToPILImage()(x_gray.squeeze(0).cpu())
    gray_filepath = filepath.rsplit('.', 1)[0] + '.' + 'gray' + '.' + filepath.rsplit('.', 1)[1]
    x_gray_pil.save(gray_filepath)

    for detector in detectors:
        if detector == 'dx':
            grads = K.filters.spatial_gradient(x_gray, order=1)
            grads_x = grads[:, :, 0]
            output = K.utils.tensor_to_image(1. - grads_x.clamp(0., 1.))

        elif detector == 'dy':
            grads = K.filters.spatial_gradient(x_gray, order=1)
            grads_y = grads[:, :, 1]
            output = K.utils.tensor_to_image(1. - grads_y.clamp(0., 1.))

        elif detector == 'd2x':
            grads = K.filters.spatial_gradient(x_gray, order=2)
            grads_x = grads[:, :, 0]
            output = K.utils.tensor_to_image(1. - grads_x.clamp(0., 1.))

        elif detector == 'd2y':
            grads = K.filters.spatial_gradient(x_gray, order=2)
            grads_y = grads[:, :, 1]
            output = K.utils.tensor_to_image(1. - grads_y.clamp(0., 1.))

        elif detector == 'sobel':
            x_sobel = K.filters.sobel(x_gray)
            print(f"Sobel filter output min: {x_sobel.min()}, max: {x_sobel.max()}")
            x_sobel = (x_sobel - x_sobel.min()) / (x_sobel.max() - x_sobel.min())
            output = K.utils.tensor_to_image(x_sobel)

        elif detector == 'laplacian':
            x_laplacian = K.filters.laplacian(x_gray, kernel_size=5)
            print(f"Laplacian filter output min: {x_laplacian.min()}, max: {x_laplacian.max()}")
            x_laplacian = (x_laplacian - x_laplacian.min()) / (x_laplacian.max() - x_laplacian.min())
            output = K.utils.tensor_to_image(x_laplacian)

        elif detector == 'icanny':
            x_icanny = K.filters.canny(x_gray)[0]
            output = K.utils.tensor_to_image(x_icanny.clamp(0., 1.0))  # No inversion here.

        elif detector == 'canny':
            x_canny = K.filters.canny(x_gray)[0]
            output = K.utils.tensor_to_image(1. - x_canny.clamp(0., 1.0))

        output_filepath = filepath.rsplit('.', 1)[0] + '.' + detector + '.' + filepath.rsplit('.', 1)[1]

        output = output.astype('float32')  # Make sure the output is float for the calculations
        output = output - output.min()  # shift values to be >= 0
        output = output / output.max()  # normalize to range [0, 1]
        output = (output * 255).astype('uint8')  # scale to [0, 255]

        print(f"Scaled output min: {output.min()}, max: {output.max()}")
        cv2.imwrite(output_filepath, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

def process_file(args):
    """
    Helper function to process a single file.
    """
    filepath, detectors = args
    edge_detection(filepath, detectors)


def split_string(s):
    """
    Helper function to split input string into list by commas and check the inputs are valid.
    """
    valid_choices = ["dx", "dy", "d2x", "d2y", "sobel", "laplacian", "canny", "icanny"]
    input_list = s.split(',')
    
    # Check that each input is a valid choice
    for input_choice in input_list:
        if input_choice not in valid_choices:
            raise argparse.ArgumentTypeError(f"Invalid choice: {input_choice} (choose from {valid_choices})")

    return input_list

def main():
    """
    Main function to parse command line arguments and start edge detection.
    """
    parser = argparse.ArgumentParser(description='Edge detection on an image.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-i', '--image', help='Path of the single image file.')
    group.add_argument('-r', '--recursive', help='Recursive file search with file filter, e.g., "*.jpg".')
    parser.add_argument('-d', '--detectors', required=True, help='Edge detectors to use (comma-separated).',
                        type=split_string)  # Remove choices parameter here
    args = parser.parse_args()

    if args.image:
        edge_detection(args.image, args.detectors)
    elif args.recursive:
        file_filter = os.path.join(os.getcwd(), '**', args.recursive)
        filepaths = glob.glob(file_filter, recursive=True)

        with Pool() as pool:
            pool.map(process_file, [(filepath, args.detectors) for filepath in filepaths])

if __name__ == '__main__':
    main()
