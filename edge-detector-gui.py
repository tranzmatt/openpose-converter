import gradio as gr
import kornia as K
from kornia.core import Tensor

def edge_detection(filepath, detector, low_threshold=0, high_threshold=255, sigma=1, kernel_size=5, hysteresis=True, eps=1e-6, blur_sigma=0):
    img: Tensor = K.io.load_image(filepath, K.io.ImageLoadType.RGB32)
    img = img[None]
    x_gray = K.color.rgb_to_grayscale(img)

    if detector == '1st order derivates in x':
        grads: Tensor = K.filters.spatial_gradient(x_gray, order=1)
        grads_x = grads[:, :, 0]
        output = K.utils.tensor_to_image(1. - grads_x.clamp(0., 1.))

    elif detector == '1st order derivates in y':
        grads: Tensor = K.filters.spatial_gradient(x_gray, order=1)
        grads_y = grads[:, :, 1]
        output = K.utils.tensor_to_image(1. - grads_y.clamp(0., 1.))

    elif detector == '2nd order derivatives in x':
        grads: Tensor = K.filters.spatial_gradient(x_gray, order=2)
        grads_x = grads[:, :, 0]
        output = K.utils.tensor_to_image(1. - grads_x.clamp(0., 1.))

    elif detector == '2nd order derivatives in y':
        grads: Tensor = K.filters.spatial_gradient(x_gray, order=2)
        grads_y = grads[:, :, 1]
        output = K.utils.tensor_to_image(1. - grads_y.clamp(0., 1.))

    elif detector == 'Sobel':
        x_sobel: Tensor = K.filters.sobel(x_gray)
        output = K.utils.tensor_to_image(1. - x_sobel)

    elif detector == 'Laplacian':
        x_laplacian: Tensor = K.filters.laplacian(x_gray, kernel_size=5)
        output = K.utils.tensor_to_image(1. - x_laplacian.clamp(0., 1.))

    elif detector == 'Canny' or detector == 'iCanny':
        low_threshold /= 255.0
        high_threshold /= 255.0
        sigma = (sigma, sigma)
        kernel_size = (kernel_size, kernel_size)
        
        # Apply Gaussian blur if blur_sigma is not 0
        if blur_sigma > 0:
            blur_size = (int(blur_sigma*4)+1, int(blur_sigma*4)+1)  # kernel size is chosen as 4*sigma + 1 to cover 95% of the gaussian distribution
            x_gray = K.filters.gaussian_blur2d(x_gray, blur_size, (blur_sigma, blur_sigma))

        x_canny: Tensor = K.filters.canny(x_gray, low_threshold=low_threshold, high_threshold=high_threshold, sigma=sigma, kernel_size=kernel_size, hysteresis=hysteresis, eps=eps)[0]
        
        if detector == 'Canny':
            output = K.utils.tensor_to_image(1. - x_canny.clamp(0., 1.0))
        else:
            output = K.utils.tensor_to_image(x_canny.clamp(0., 1.0))

    return output


examples = [
    ["examples/huggingface.jpg", "1st order derivates in x"],
    ["examples/doraemon.jpg", "Canny"]
]

title = "Kornia Edge Detection"
description = "<p style='text-align: center'>This is a Gradio interface for Kornia's Edge Detection.</p><p style='text-align: center'>To use it, simply upload your image, or click one of the examples to load them, and select any edge detector to run it! Read more at the links at the bottom.</p>"
article = "<p style='text-align: center'><a href='https://kornia.readthedocs.io/en/latest/' target='_blank'>Kornia Docs</a> | <a href='https://github.com/kornia/kornia' target='_blank'>Kornia Github Repo</a> | <a href='https://kornia-tutorials.readthedocs.io/en/latest/filtering_edges.html' target='_blank'>Kornia Edge Detection Tutorial</a></p>"

iface = gr.Interface(
    edge_detection,
    [
        gr.Image(type="filepath"),
        gr.inputs.Dropdown(choices=["1st order derivates in x", "1st order derivates in y", "2nd order derivatives in x", "2nd order derivatives in y", "Sobel", "Laplacian", "Canny", "iCanny"]),
        gr.inputs.Slider(minimum=1, maximum=255, step=1, default=100, label="Low Threshold"), # For canny_params["low_threshold"]
        gr.inputs.Slider(minimum=1, maximum=255, step=1, default=200, label="High Threshold"), # For canny_params["high_threshold"]
        gr.inputs.Slider(minimum=0, maximum=10, step=0.1, default=1, label="Sigma"), # For canny_params["sigma"]
        gr.inputs.Slider(minimum=1, maximum=31, step=2, default=5, label="Kernel Size"), # For canny_params["kernel_size"]
        gr.inputs.Checkbox(label="Hysteresis"), # For canny_params["hysteresis"]
        gr.inputs.Number(default=1e-6, label="Eps"), # For canny_params["eps"]
        gr.inputs.Slider(minimum=0, maximum=10, step=0.1, default=0, label="Blur Sigma"), # For canny_params["blur_sigma"]
    ],
    "image",
    examples,
    title=title,
    description=description,
    article=article
)

iface.launch(server_name="0.0.0.0")

