import random

import numpy as np
from skimage.transform import rotate
from skimage.util import random_noise
from skimage import exposure
from skimage.filters import gaussian

augmentation_functions = {
    'rotate': lambda image: rotate(image, random.randint(0, 360)),
    'noise': lambda image: random_noise(image),
    'gamma': lambda image: exposure.adjust_gamma(image, random.uniform(0.5, 1.5)),
    'gaussian': lambda image: gaussian(image),
}

def augment_images(
    images, 
    rotation_factor=0.05,
    noise_factor=0.05,
    gamma_factor=0.05,
    gaussian_factor=0.05,
):
    augmentation_rates = [rotation_factor, noise_factor, gamma_factor, gaussian_factor]
    rates = sum(augmentation_rates)

    augmented_images = []
    for (image_dir, image, label) in images:
        if random.random() < rates:
            function = random.choices(
                list(augmentation_functions.keys()), 
                weights=[rotation_factor, noise_factor, gamma_factor, gaussian_factor]
            )[0]
            augmented_image = augmentation_functions[function](image)
            min_image, max_image = np.min(augmented_image), np.max(augmented_image)
            augmented_image = ((augmented_image - min_image) / (max_image - min_image)) * 255
            augmented_image = augmented_image.astype('uint8')
            augmented_images.append((image_dir, augmented_image, label))
    return augmented_images