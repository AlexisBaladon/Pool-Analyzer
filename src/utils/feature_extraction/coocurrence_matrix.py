from typing import Callable
from skimage.feature import graycomatrix, graycoprops

def create_coocurrence_matrix_features(images: list[tuple], 
                                       features_to_extract: list, 
                                       to_grayscale: Callable, 
                                       distance, angle, levels=256):
    pixels_df = {'image_id': [], 
                 'feature_name': [], 
                 'feature_value': [], 
                 'label': []}
    single_images = [image for _, image, _ in images]
    grayscale_images = to_grayscale(single_images)
    grayscale_images = [(id, image, label) 
                        for (id, _, label), image in 
                        zip(images, grayscale_images)]

    for id, image, label in grayscale_images:
        co_occurrence_matrix = graycomatrix(image, 
                                            [distance], 
                                            [angle], 
                                            levels=levels, 
                                            normed=True,
                                            symmetric=True,)
        for feature in features_to_extract:
            feature_value = graycoprops(co_occurrence_matrix, feature)[0, 0]
            pixels_df['image_id'].append(id)
            pixels_df['feature_value'].append(feature_value)
            pixels_df['feature_name'].append(feature + '_coocurrence_matrix')
            pixels_df['label'].append(label)

    pixels_df['feature_value'] = list(map(float, pixels_df['feature_value']))
    return pixels_df