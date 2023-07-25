#!/usr/bin/env python3

import gradio as gr
import cv2
import numpy as np

def canny_edge_detector(input_img, low_threshold, high_threshold):
    # Convert the input image into grayscale
    grayscaled_image = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)

    # Apply Canny edge detection
    canny_image = cv2.Canny(grayscaled_image, int(low_threshold), int(high_threshold))

    # Normalize the result for visualization
    canny_image = cv2.normalize(canny_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return canny_image

demo = gr.Interface(
    fn=canny_edge_detector,
    inputs=[
        gr.inputs.Image(),
        gr.inputs.Slider(minimum=0, maximum=255, default=100, label="Low threshold"),
        gr.inputs.Slider(minimum=0, maximum=255, default=200, label="High threshold"),
    ],
    outputs="image"
)

demo.launch(server_name="0.0.0.0")
