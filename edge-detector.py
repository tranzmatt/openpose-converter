#!/usr/bin/env python3

"""
Edge detection script using Kornia library.
"""
import argparse
import os
import glob
from multiprocessing import Pool
import torch
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

    #img = cv2.imread(filepath)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    #img: Tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # permute dimensions and normalize
    #img = img[None].to(device)

    img = K.io.load_image(filepath, K.io.ImageLoadType.RGB32).to(device)
    img = img[None]
    x_gray = K.color.rgb_to_grayscale(img)

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
            output = K.utils.tensor_to_image(1. - x_sobel)

        elif detector == 'laplacian':
            x_laplacian = K.filters.laplacian(x_gray, kernel_size=5)
            output = K.utils.tensor_to_image(1. - x_laplacian.clamp(0., 1.))

        elif detector == 'canny':
            x_canny = K.filters.canny(x_gray)[0]
            output = K.utils.tensor_to_image(1. - x_canny.clamp(0., 1.0))

        output_filepath = filepath.rsplit('.', 1)[0] + '_' + detector + '.' + filepath.rsplit('.', 1)[1]
        #output.save(output_filepath)
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
    valid_choices = ["dx", "dy", "d2x", "d2y", "sobel", "laplacian", "canny"]
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
