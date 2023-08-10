#!/usr/bin/env python3

import os
import sys
import argparse
import glob
import cv2

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

# Detector functions
def canny(img, l, h):
    global model_canny
    if model_canny is None:
        from annotator.canny import CannyDetector
        model_canny = CannyDetector()
    result = model_canny(img, l, h)
    return result

def hed(img):
    global model_hed
    if model_hed is None:
        from annotator.hed import HEDdetector
        model_hed = HEDdetector()
    result = model_hed(img)
    return result

def mlsd(img, thr_v, thr_d):
    global model_mlsd
    if model_mlsd is None:
        from annotator.mlsd import MLSDdetector
        model_mlsd = MLSDdetector()
    result = model_mlsd(img, thr_v, thr_d)
    return result

def midas(img, a):
    global model_midas
    if model_midas is None:
        from annotator.midas import MidasDetector
        model_midas = MidasDetector()
    result = model_midas(img, a)
    return result

def openpose(img, has_hand):
    global model_openpose
    if model_openpose is None:
        from annotator.openpose import OpenposeDetector
        model_openpose = OpenposeDetector()
    result, _ = model_openpose(img, has_hand)
    return result

def uniformer(img):
    global model_uniformer
    if model_uniformer is None:
        from annotator.uniformer import UniformerDetector
        model_uniformer = UniformerDetector()
    result = model_uniformer(img)
    return result

def process_image(img_path, args):
    img = cv2.imread(img_path)
    img = resize_image(HWC3(img), args.resolution)
    
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    output_dir = os.path.dirname(img_path)
    output_name = f"{base_name}_resize{args.resolution}.png"
    cv2.imwrite(os.path.join(output_dir, output_name), img)
    
    if args.canny:
        result = canny(img, args.canny_low, args.canny_high)
        print(f"Canny - Input dimensions: {img.shape}, Result dimensions: {result.shape}")
        output_name = f"{base_name}_canny_res{args.resolution}_low{args.canny_low}_high{args.canny_high}.png"
        cv2.imwrite(os.path.join(output_dir, output_name), result)
    
    if args.hed:
        result = hed(img)
        output_name = f"{base_name}_hed_res{args.resolution}.png"
        cv2.imwrite(os.path.join(output_dir, output_name), result)
    
    if args.mlsd:
        result = mlsd(img, args.mlsd_value, args.mlsd_distance)
        output_name = f"{base_name}_mlsd_res{args.resolution}_v{args.mlsd_value}_d{args.mlsd_distance}.png"
        cv2.imwrite(os.path.join(output_dir, output_name), result)
    
    if args.midas:
        result = midas(img, args.midas_alpha)
        output_name = f"{base_name}_midas_res{args.resolution}_alpha{args.midas_alpha}.png"
        cv2.imwrite(os.path.join(output_dir, output_name), result)
    
    if args.openpose:
        result = openpose(img, args.pose_hand)
        hand_suffix = "_hand" if args.pose_hand else ""
        output_name = f"{base_name}_openpose_res{args.resolution}{hand_suffix}.png"
        cv2.imwrite(os.path.join(output_dir, output_name), result)
    
    if args.uniformer:
        result = uniformer(img)
        output_name = f"{base_name}_uniformer_res{args.resolution}.png"
        cv2.imwrite(os.path.join(output_dir, output_name), result)

# Main function
def main():
    parser = argparse.ArgumentParser(description="Batch command line version of the Gradio application.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the input image or directory.")
    parser.add_argument("-r", "--resolution", type=int, default=512, help="Resolution for processing. Default is 512.")

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

    # File gathering logic
    valid_extensions = ['*.jpg', '*.JPG', '*.png', '*.PNG']
    image_files = []
    if os.path.isdir(args.input):
        for ext in valid_extensions:
            image_files.extend(glob.glob(os.path.join(args.input, '**', ext), recursive=True))
        for img_path in image_files:
            process_image(img_path, args)
    else:
        process_image(args.input, args)

if __name__ == "__main__":
    main()

