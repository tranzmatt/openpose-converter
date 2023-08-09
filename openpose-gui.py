#!/usr/bin/env python3

import gradio as gr
import cv2
import numpy as np

from ControlNet.annotator import util
from ControlNet.annotator.openpose.model import bodypose_model

def openpose_detector(input_img, low_threshold, high_threshold):
    # Convert the input image into grayscale
    grayscaled_image = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)
    body_estimation = Body('../model/body_pose_model.pth')
    candidate, subset = body_estimation(input_img)

    canvas = np.zeros_like(input_img)
    canvas.fill(0)
    canvas = util.draw_bodypose(canvas, candidate, subset)

    return openpose_image

demo = gr.Interface(
    fn=openpose_detector,
    inputs=[
        gr.inputs.Image(),
        gr.inputs.Slider(minimum=0, maximum=255, default=100, label="Low threshold"),
        gr.inputs.Slider(minimum=0, maximum=255, default=200, label="High threshold"),
    ],
    outputs="image"
)

demo.launch(server_name="0.0.0.0")
