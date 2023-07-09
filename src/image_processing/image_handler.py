import numpy as np
from PIL import Image

def load_image(image_path):
    image = np.array(Image.open(image_path))
    return image

def to_grayscale(images: list):
    grayscale_images = []
    for image in images:
        grayscale_image = np.array(Image.fromarray(image).convert('L'))
        grayscale_images.append(grayscale_image)
    return grayscale_images

def to_histogram(image_gray: np.ndarray, bins: int = 256):
    hist, _ = np.histogram(image_gray, bins=bins, density=True)
    return hist