#!/usr/bin/env python3

import os
import sys
import argparse
import glob
import cv2
import numpy as np

import hashlib

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_dir)
sys.path.append(os.path.join(script_dir, 'ControlNet'))

from annotator.util import resize_image, HWC3

# Global model initializations
model_canny = None
model_hed = None
model_mlsd = None
model_midas = None
model_openpose = None
model_uniformer = None


DETECTOR_NAMES = ["canny", "hed", "mlsd", "midas", "openpose", "uniformer"]

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

# Detector functions
def canny(img, l, h):
    global model_canny
    if model_canny is None:
        from ControlNet.annotator.canny import CannyDetector
        model_canny = CannyDetector()
    result = model_canny(img, l, h)
    return result

def hed(img):
    global model_hed
    if model_hed is None:
        from ControlNet.annotator.hed import HEDdetector
        model_hed = HEDdetector()
    result = model_hed(img)
    return result

def mlsd(img, thr_v, thr_d):
    global model_mlsd
    if model_mlsd is None:
        from ControlNet.annotator.mlsd import MLSDdetector
        model_mlsd = MLSDdetector()
    result = model_mlsd(img, thr_v, thr_d)
    return result

def midas(img, a):
    global model_midas
    if model_midas is None:
        from ControlNet.annotator.midas import MidasDetector
        model_midas = MidasDetector()
    result = model_midas(img, a)
    return result

def openpose(img, has_hand):
    global model_openpose
    if model_openpose is None:
        from ControlNet.annotator.openpose import OpenposeDetector
        model_openpose = OpenposeDetector()
    result, _ = model_openpose(img, has_hand)
    return result

def uniformer(img):
    global model_uniformer
    if model_uniformer is None:
        from ControlNet.annotator.uniformer import UniformerDetector
        model_uniformer = UniformerDetector()
    result = model_uniformer(img)
    return result

def process_image(img_path, args):
    print(f"Processing {img_path}")
    img = cv2.imread(img_path)
    the_hash = hashlib.md5(img.tobytes()).hexdigest()
    print(f"Original hash is {the_hash}")
    img = resize_image(HWC3(img), args.resolution)
    
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    output_dir = os.path.dirname(img_path)
    output_name = f"{base_name}_resize{args.resolution}.png"
    cv2.imwrite(os.path.join(output_dir, output_name), img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    the_hash = hashlib.md5(img.tobytes()).hexdigest()
    print(f"Resize hash is {the_hash}")
    
    if args.canny:
        result = canny(img, args.canny_low, args.canny_high)
        output_name = f"{base_name}_canny_resize{args.resolution}_low{args.canny_low}_high{args.canny_high}.png"
        cv2.imwrite(os.path.join(output_dir, output_name), result)
    
    if args.hed:
        result = hed(img)
        output_name = f"{base_name}_hed_resize{args.resolution}.png"
        cv2.imwrite(os.path.join(output_dir, output_name), result)
    
    if args.mlsd:
        result = mlsd(img, args.mlsd_value, args.mlsd_distance)
        output_name = f"{base_name}_mlsd_resize{args.resolution}_v{args.mlsd_value}_d{args.mlsd_distance}.png"
        cv2.imwrite(os.path.join(output_dir, output_name), result)
    
    if args.openpose:
        result = openpose(img, args.pose_hand)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
        # Check if the result is an all-black image
        if np.all(result == 0):
            print(f"No openpose could be detected for {img_path}.")
        else:
            hand_suffix = "_hand" if args.pose_hand else ""
            output_name = f"{base_name}_openpose{hand_suffix}_res{args.resolution}.png"
            cv2.imwrite(os.path.join(output_dir, output_name), result)

    if args.midas:
        depth_result, normal_result = midas(img, args.midas_alpha)
        depth_output_name = f"{base_name}_midas_depth_res{args.resolution}_alpha{args.midas_alpha}.png"
        normal_output_name = f"{base_name}_midas_normal_res{args.resolution}_alpha{args.midas_alpha}.png"
        normal_result = cv2.cvtColor(normal_result, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(output_dir, depth_output_name), depth_result)
        cv2.imwrite(os.path.join(output_dir, normal_output_name), normal_result)

    if args.uniformer:
        primary_result = uniformer(img)
        primary_result = cv2.cvtColor(primary_result, cv2.COLOR_BGR2RGB)
        primary_output_name = f"{base_name}_uniformer_res{args.resolution}.png"
        cv2.imwrite(os.path.join(output_dir, primary_output_name), primary_result)

# Main function
def main():
    parser = argparse.ArgumentParser(description="Batch command line version of the Gradio application.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the input image or directory.")
    parser.add_argument("-r", "--resolution", type=int, default=512, help="Resolution for processing. Default is 512.")
    parser.add_argument("-f", "--filter", nargs='*', help="Input filter(s) for processing images in a directory. E.g. '*.rembg.jpg'")

    # Canny specific arguments
    parser.add_argument("-c", "--canny", action="store_true", help="Use Canny detector.")
    parser.add_argument("-cl", "--canny_low", type=int, default=100, help="Canny low threshold. Default is 100.")
    parser.add_argument("-ch", "--canny_high", type=int, default=200, help="Canny high threshold. Default is 200.")

    # HED specific arguments
    parser.add_argument("-e", "--hed", action="store_true", help="Use HED detector.")

    # MLSD specific arguments
    parser.add_argument("-m", "--mlsd", action="store_true", help="Use MLSD detector.")
    parser.add_argument("-mv", "--mlsd_value", type=float, default=0.1, help="MLSD value threshold. Default is 0.1.")
    parser.add_argument("-md", "--mlsd_distance", type=float, default=0.1, help="MLSD distance threshold. Default is 0.1.")

    # MIDAS specific arguments
    parser.add_argument("-d", "--midas", action="store_true", help="Use MIDAS detector.")
    parser.add_argument("-ma", "--midas_alpha", type=float, default=6.2, help="MIDAS alpha value. Default is 6.2.")

    # Openpose specific arguments
    parser.add_argument("-p", "--openpose", action="store_true", help="Use Openpose detector.")
    parser.add_argument("-ph", "--pose_hand", action="store_true", help="Detect hand in Openpose. Default is False.")

    # Uniformer specific arguments
    parser.add_argument("-u", "--uniformer", action="store_true", help="Use Uniformer detector.")

    args = parser.parse_args()
    image_files = get_image_files(args.input, args.filter)

    for img_path in image_files:
        process_image(img_path, args)

if __name__ == "__main__":
    main()

