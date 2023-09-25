from typing import Callable

import scipy
import numpy as np

def calculate_mean(image_channel: np.ndarray):
    return image_channel.mean()

def calculate_median(image_channel: np.ndarray):
    return np.median(image_channel)

def calculate_mode(image_channel: np.ndarray):
    counts = [0] * 256
    for row in image_channel:
        for pixel in row:
            counts[pixel] += 1
    return np.argmax(counts)

def calculate_std(image_channel: np.ndarray):
    return image_channel.std()

def calculate_min(image_channel: np.ndarray):
    return image_channel.min()

def calculate_max(image_channel: np.ndarray):
    return image_channel.max()

def calculate_range(image_channel: np.ndarray):
    return image_channel.max() - image_channel.min()

def calculate_quantile(image_channel: np.ndarray, q: float):
    bin_count = 256
    hist, bin_edges = np.histogram(image_channel, bins=bin_count)
    cum_values = np.cumsum(hist * np.diff(bin_edges))
    quantile = np.interp(q * 100, cum_values, bin_edges[:-1])
    return quantile

def calculate_iqr(image_channel: np.ndarray):
    return calculate_quantile(image_channel, 0.75) - calculate_quantile(image_channel, 0.25)

def entropy(image_channel: np.ndarray):
    hist, _ = np.histogram(image_channel, bins=256)
    hist = hist[hist != 0]
    prob = hist / hist.sum()
    return -np.sum(prob * np.log2(prob))

def calculate_skewness(image_channel: np.ndarray):
    return scipy.stats.skew(image_channel, axis=None)

def calculate_kurtosis(image_channel: np.ndarray):
    return scipy.stats.kurtosis(image_channel, axis=None)

channel_feature_functions = {
    'mean': calculate_mean,
    'std': calculate_std,
    'median': calculate_median,
    'mode': calculate_mode,
    'min': calculate_min,
    'max': calculate_max,
    'range': calculate_range,
    'skewness': calculate_skewness,
    'kurtosis': calculate_kurtosis,
    'entropy': entropy,
    'quantile_0.25': lambda img: calculate_quantile(img, q=0.25),
    'quantile_0.75': lambda img: calculate_quantile(img, q=0.75),
    'iqr': calculate_iqr,
}

def create_channel_features(images: list, 
                            features_to_extract: list, 
                            to_grayscale: Callable = None):
    channels = ['red', 'green', 'blue'] + (['grayscale'] if to_grayscale is not None else [])
    pixels_df = {'image_id': [], 'feature_name': [], 'feature_value': [], 'label': []}
    selected_feature_functions = {feature_name: channel_feature_functions[feature_name] 
                                  for feature_name in features_to_extract}

    for id, image, label in images:
        for feature_name, feature_function in selected_feature_functions.items():
            for chnl in range(len(channels)):
                if channels[chnl] == 'grayscale':
                    if to_grayscale is None:
                        continue
                    extraction_image, = to_grayscale([image])
                else:
                    extraction_image = image[:, :, chnl]

                pixels_df['image_id'].append(id)
                pixels_df['feature_value'].append(feature_function(extraction_image))
                pixels_df['feature_name'].append(feature_name + '_' + channels[chnl])
                pixels_df['label'].append(label)

    pixels_df['feature_value'] = list(map(float, pixels_df['feature_value']))
    return pixels_df