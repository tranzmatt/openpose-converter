#!/usr/bin/env python3
# This code from https://github.com/lllyasviel/ControlNet

import os
import sys
import argparse
import cv2
import numpy as np
import math
import glob
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import torch

import urllib
import json

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_dir)
sys.path.append(os.path.join(script_dir, 'ControlNet'))

import ControlNet.annotator.openpose.util as util
from ControlNet.annotator.openpose.body import Body
from ControlNet.annotator.openpose.hand import Hand


def get_pose_json(ori_img):
    height, width, channels = ori_img.shape

    candidate, subset = body_estimation(ori_img)

    if len(candidate) == 0 or len(subset) == 0:
        print("No poses found in the input image.")
        return None

    candidate_int = candidate.astype(int)
    candidate_list = candidate_int[:, :2].tolist()

    data = {
        "width": width,
        "height": height,
        "keypoints": candidate_list
    }
    candidate_json = json.dumps(data)

    return candidate_json


def process_image(this_input_image, the_body_estimation, the_hand_estimation, these_args):
    print(f"Examining {this_input_image}")
    output_png_filename = '.'.join(this_input_image.split('.')[:-1]) + '.openpose.png'
    output_json_filename = '.'.join(this_input_image.split('.')[:-1]) + '.openpose.json'

    if os.path.isfile(output_png_filename) and not args.force:
        print(f"Output file {output_png_filename} already exists, skipping {this_input_image}")
        return

    ori_img = cv2.imread(this_input_image)  # B,G,R order

    canvas = np.zeros_like(ori_img)
    canvas.fill(0)

    try:
        print(f"Estimating Body")
        body_candidate, body_subset = the_body_estimation(ori_img)
        if len(body_candidate) == 0 or len(body_subset) == 0:
            print(f"No poses found in the input image {this_input_image}.")
            return
        canvas = util.draw_bodypose(canvas, body_candidate, body_subset)
    except Exception as e:
        print(f"Error processing image body {this_input_image}: {e}")
        return

    try:
        peaks = the_hand_estimation(ori_img)
        print(f"Estimating Hand {peaks}")
        if peaks is not None:
            canvas = util.draw_handpose(canvas, peaks)
    except Exception as e:
        print(f"Error processing image hand {this_input_image}: {e}")
        return

    cv2.imwrite(output_png_filename, canvas)

    if these_args.json_output:
        candidate_json = get_pose_json(ori_img)
        with open(output_json_filename, 'w') as f:
            f.write(candidate_json)

    if these_args.show_image:
        plt.imshow(canvas[:, :, [2, 1, 0]])
        plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Body Pose Estimation using OpenPose', add_help=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-i", "--input_image", help="Path to the input image", type=str)
    group.add_argument("-d", "--directory", help="Directory to search for images", type=str)
    parser.add_argument("-p", "--patterns", help="Pattern to match for images in directory", type=str)
    parser.add_argument("-r", "--recursive", help="Search for files in subdirectories recursively",
                        action="store_true")
    parser.add_argument("-s", "--show_image", help="Display the output image", action="store_true")
    parser.add_argument("-j", "--json_output", help="Save JSON output to file", action="store_true")
    parser.add_argument("-f", "--force", help="Force processing of images even if output file already exists",
                        action="store_true")
    args = parser.parse_args()

    # Print help and quit if -h or --help is provided
    if args.input_image is None and args.directory is None:
        parser.print_help()
        exit()

    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)

    model_dir = os.path.join(script_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    body_model_file = os.path.join(model_dir, "body_pose_model.pth")

    if not os.path.isfile(body_model_file):
        body_model_path = \
            "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/body_pose_model.pth"
        urllib.request.urlretrieve(body_model_path, body_model_file)

    body_estimation = Body(body_model_file)

    hand_model_file = os.path.join(model_dir, "hand_pose_model.pth")

    if not os.path.isfile(hand_model_file):
        hand_model_path = \
            "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/hand_pose_model.pth"
        urllib.request.urlretrieve(hand_model_path, hand_model_file)
    hand_estimation = Hand(hand_model_file)

    if args.input_image:
        process_image(args.input_image, body_estimation, hand_estimation, args)
    else:
        patterns = args.patterns.split(',')
        for pattern in patterns:
            for input_image in glob.iglob(os.path.join(args.directory, '**', pattern)
                                          if args.recursive else os.path.join(args.directory, pattern),
                                          recursive=args.recursive):
                print(f"Processing {input_image}")
                process_image(input_image, body_estimation, hand_estimation, args)
