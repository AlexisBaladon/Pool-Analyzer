from typing import Callable

import numpy as np

def calculate_mean_histogram(histogram: np.ndarray, numBins: int):
    return np.sum((np.arange(numBins) / numBins) * histogram)

def calculate_std_histogram(histogram: np.ndarray, numBins: int):
    mean = calculate_mean_histogram(histogram, numBins)
    return np.sqrt(np.sum(((np.arange(numBins) / numBins) - mean) ** 2 * histogram))

def calculate_median_histogram(histogram: np.ndarray, _: int):
    cumsum = np.cumsum(histogram)
    return np.argmax(cumsum >= cumsum[-1] / 2)

def calculate_mode_histogram(histogram: np.ndarray, _: int):
    return np.argmax(histogram)

def calculate_min_histogram(histogram: np.ndarray, _: int):
    return np.argmin(histogram)

def calculate_max_histogram(histogram: np.ndarray, _: int):
    return np.argmax(histogram)

def calculate_range_histogram(histogram: np.ndarray, _: int):
    return np.argmax(histogram) - np.argmin(histogram)

def calculate_quantile_histogram(histogram: np.ndarray, numBins: int, q: float):
    cum_values = np.cumsum(histogram)
    quantile = np.interp(q * 100, cum_values, np.arange(numBins))
    return quantile

def calculate_iqr_histogram(histogram: np.ndarray, numBins: int):
    return calculate_quantile_histogram(histogram, numBins, q=0.75) - calculate_quantile_histogram(histogram, numBins, q=0.25)

def calculate_skewness_histogram(histogram: np.ndarray, numBins: int):
    mean = calculate_mean_histogram(histogram, numBins)
    skewness = np.sum(((histogram - mean)**3) * histogram)
    return skewness

def calculate_kurtosis_histogram(histogram: np.ndarray, numBins: int):
    mean = calculate_mean_histogram(histogram, numBins)
    kurtosis = np.sum(((histogram - mean)**4) * histogram)
    return kurtosis

def calculate_uniformity_histogram(histogram: np.ndarray, _: int):
    return np.sum(histogram ** 2)

def calculate_entropy_histogram(histogram: np.ndarray, numBins: int):
    entropy = -np.sum(histogram * np.log2(histogram + 1e-12)) / np.log2(numBins)
    return entropy

def calculate_R_histogram(histogram: np.ndarray, _: int):
    R = 1 - 1 / (1 + np.std(histogram)**2)
    return R

histogram_feature_functions = {'mean': calculate_mean_histogram,
                               'std': calculate_std_histogram,
                               'median': calculate_median_histogram,
                               'mode': calculate_mode_histogram,
                               'min': calculate_min_histogram,
                               'max': calculate_max_histogram,
                               'range': calculate_range_histogram,
                               'skewness': calculate_skewness_histogram,
                               'kurtosis': calculate_kurtosis_histogram,
                               'uniformity': calculate_uniformity_histogram,
                               'entropy': calculate_entropy_histogram,
                               'R': calculate_R_histogram}

def create_histogram_features(images: list[tuple], 
                              features_to_extract: list, 
                              to_grayscale: Callable, 
                              to_histogram: Callable, 
                              bins=256):
    pixels_df = {'image_id': [], 
                 'feature_name': [], 
                 'feature_value': [], 
                 'label': []}
    selected_features = {feature_name: histogram_feature_functions[feature_name] 
                         for feature_name in features_to_extract}

    for id, image, label in images:
        grayscale_image = to_grayscale(image)
        image_histogram = to_histogram(grayscale_image, bins=bins)

        for feature_name, feature_function in selected_features.items():
            feature_value = feature_function(image_histogram, bins)
            pixels_df['image_id'].append(id)
            pixels_df['feature_value'].append(feature_value)
            pixels_df['feature_name'].append(feature_name + '_histogram')
            pixels_df['label'].append(label)

    pixels_df['feature_value'] = list(map(float, pixels_df['feature_value']))
    return pixels_df