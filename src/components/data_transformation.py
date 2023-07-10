import dataclasses
from typing import Callable

from ..feature_extraction import \
    channel_features, histogram_features, coocurrence_matrix, gabor

import pandas as pd

@dataclasses.dataclass
class DataTransformationConfig:
    def __init__(self, 
        channel_features: list, 
        histogram_features: list, 
        coocurrence_matrix_features: list,
        positive_class: str,
        negative_class: str,
        to_grayscale: Callable,
        to_histogram: Callable,
    ):
        self.channel_features = channel_features
        self.histogram_features = histogram_features
        self.coocurrence_matrix_features = coocurrence_matrix_features
        self.positive_class = positive_class
        self.negative_class = negative_class
        self.to_grayscale = to_grayscale
        self.to_histogram = to_histogram

    def __repr__(self):
        return f'DataTransformationConfig(channel_features={self.channel_features}, histogram_features={self.histogram_features}, coocurrence_matrix_features={self.coocurrence_matrix_features}, to_grayscale={self.to_grayscale}, to_histogram={self.to_histogram})'
    
    def __str__(self):
        return self.__repr__()

class DataTransformer:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def __set_features_as_columns(self, features: dict) -> dict:
        feature_names = list(set(features['feature_name']))
        features_df = {feature_name: [] for feature_name in feature_names}
        features_df.update({'label': [], 'image_id': []})
        last_image_id = None

        for feature_name, feature_value, image_id, label in zip(features['feature_name'], features['feature_value'], features['image_id'], features['label']):
            if last_image_id != image_id:
                last_image_id = image_id
                features_df['label'].append(label)
                features_df['image_id'].append(image_id)
            features_df[feature_name].append(feature_value)

        return features_df

    def __transform_labels(self, feature_df: pd.DataFrame, positive_class: str, negative_class: str):
        label_to_int = lambda label: 1 if label == positive_class else 0 if label == negative_class else None
        feature_df.loc[:, 'label'] = feature_df['label'].apply(label_to_int)
        return feature_df

    def __transform_split(
        self, 
        labeled_images: list, 
        gabor_filters: list,
        channel_features_to_extract: list,
        histogram_features_to_extract: list,
        coocurrence_matrix_features_to_extract: list,
        to_grayscale: Callable, 
        to_histogram: Callable,
        positive_class: str,
        negative_class: str,
    ) -> pd.DataFrame:
        images = [(id, gabor.apply_filter(image, gabor_filters), label) for id, image, label in labeled_images]
        split_channel_features = channel_features.create_channel_features(images, channel_features_to_extract, to_grayscale)
        split_histogram_features = histogram_features.create_histogram_features(images, histogram_features_to_extract, to_grayscale, to_histogram)
        split_coocurrence_matrix_features = coocurrence_matrix.create_coocurrence_matrix_features(images, coocurrence_matrix_features_to_extract, to_grayscale, distance=1, angle=0)

        split_features_channel = self.__set_features_as_columns(split_channel_features)
        split_features_histogram = self.__set_features_as_columns(split_histogram_features)
        split_features_coocurrence_matrix = self.__set_features_as_columns(split_coocurrence_matrix_features)

        split_features_channel_df = pd.DataFrame(split_features_channel)
        split_features_histogram_df = pd.DataFrame(split_features_histogram)
        split_features_coocurrence_matrix_df = pd.DataFrame(split_features_coocurrence_matrix)
        split_features = pd.merge(split_features_channel_df, split_features_histogram_df, on=['image_id', 'label'])
        split_features = pd.merge(split_features, split_features_coocurrence_matrix_df, on=['image_id', 'label'])

        split_features = self.__transform_labels(split_features, positive_class, negative_class)
        return split_features

    def transform(self, train_images: list, test_images: list) -> tuple[tuple[pd.DataFrame, pd.DataFrame], list[str]]:
        gabor_filters = gabor.create_gaborfilter()
        channel_features_to_extract = self.config.channel_features
        histogram_features_to_extract = self.config.histogram_features
        coocurrence_matrix_features_to_extract = self.config.coocurrence_matrix_features
        train_features = self.__transform_split(train_images, gabor_filters, channel_features_to_extract, histogram_features_to_extract, coocurrence_matrix_features_to_extract, self.config.to_grayscale, self.config.to_histogram, self.config.positive_class, self.config.negative_class)
        test_features = self.__transform_split(test_images, gabor_filters, channel_features_to_extract, histogram_features_to_extract, coocurrence_matrix_features_to_extract, self.config.to_grayscale, self.config.to_histogram, self.config.positive_class, self.config.negative_class)
        return (train_features, test_features)