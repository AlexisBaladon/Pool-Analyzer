import dataclasses
from typing import Callable

from src.utils.feature_extraction import \
    color_features, channel_features, histogram_features, coocurrence_matrix, gabor
from src.utils.image_processing import augmentation

import pandas as pd

@dataclasses.dataclass
class DataTransformationConfig:
    def __init__(self, 
        color_features: list,
        channel_features: list, 
        histogram_features: list, 
        coocurrence_matrix_features: list,
        correlated_features: list,
        drop_correlated_features: bool,
        positive_class: str,
        negative_class: str,
        use_augmentation: bool,
        to_grayscale: Callable,
        to_histogram: Callable,
    ):
        self.color_features = color_features
        self.channel_features = channel_features
        self.histogram_features = histogram_features
        self.coocurrence_matrix_features = coocurrence_matrix_features
        self.correlated_features = correlated_features
        self.drop_correlated_features = drop_correlated_features
        self.positive_class = positive_class
        self.negative_class = negative_class
        self.use_augmentation = use_augmentation
        self.to_grayscale = to_grayscale
        self.to_histogram = to_histogram

    def __repr__(self):
        return f'DataTransformationConfig(color_features={self.color_features}, channel_features={self.channel_features}, histogram_features={self.histogram_features}, coocurrence_matrix_features={self.coocurrence_matrix_features}, positive_class={self.positive_class}, negative_class={self.negative_class}, use_augmentation={self.use_augmentation}, to_grayscale={self.to_grayscale}, to_histogram={self.to_histogram})'
    
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
        color_features_to_extract: list,
        channel_features_to_extract: list,
        histogram_features_to_extract: list,
        coocurrence_matrix_features_to_extract: list,
        to_grayscale: Callable, 
        to_histogram: Callable,
        positive_class: str,
        negative_class: str,
        augmented_column: str,
        augment: bool = True,
    ) -> pd.DataFrame:

        split_color_features = color_features.create_color_features(labeled_images, color_features_to_extract)
        split_channel_features = channel_features.create_channel_features(labeled_images, channel_features_to_extract, to_grayscale)
        split_histogram_features = histogram_features.create_histogram_features(labeled_images, histogram_features_to_extract, to_grayscale, to_histogram)
        split_coocurrence_matrix_features = coocurrence_matrix.create_coocurrence_matrix_features(labeled_images, coocurrence_matrix_features_to_extract, to_grayscale, distance=1, angle=0)

        split_features_color = self.__set_features_as_columns(split_color_features)
        split_features_channel = self.__set_features_as_columns(split_channel_features)
        split_features_histogram = self.__set_features_as_columns(split_histogram_features)
        split_features_coocurrence_matrix = self.__set_features_as_columns(split_coocurrence_matrix_features)

        split_features_color_df = pd.DataFrame(split_features_color)
        split_features_channel_df = pd.DataFrame(split_features_channel)
        split_features_histogram_df = pd.DataFrame(split_features_histogram)
        split_features_coocurrence_matrix_df = pd.DataFrame(split_features_coocurrence_matrix)

        split_features = pd.merge(split_features_channel_df, split_features_histogram_df, on=['image_id', 'label'])
        split_features = pd.merge(split_features, split_features_coocurrence_matrix_df, on=['image_id', 'label'])
        split_features = pd.merge(split_features, split_features_color_df, on=['image_id', 'label'])

        split_features.loc[:, augmented_column] = [0] * len(split_features)

        if augment:
            augmented_images = augmentation.augment_images(labeled_images)
            augmented_split_color_features = color_features.create_color_features(augmented_images, color_features_to_extract)
            augmented_split_channel_features = channel_features.create_channel_features(augmented_images, channel_features_to_extract, to_grayscale)
            augmented_split_histogram_features = histogram_features.create_histogram_features(augmented_images, histogram_features_to_extract, to_grayscale, to_histogram)
            augmented_split_coocurrence_matrix_features = coocurrence_matrix.create_coocurrence_matrix_features(augmented_images, coocurrence_matrix_features_to_extract, to_grayscale, distance=1, angle=0)

            augmented_split_features_color = self.__set_features_as_columns(augmented_split_color_features)
            augmented_split_features_channel = self.__set_features_as_columns(augmented_split_channel_features)
            augmented_split_features_histogram = self.__set_features_as_columns(augmented_split_histogram_features)
            augmented_split_features_coocurrence_matrix = self.__set_features_as_columns(augmented_split_coocurrence_matrix_features)

            augmented_split_features_color_df = pd.DataFrame(augmented_split_features_color)
            augmented_split_features_channel_df = pd.DataFrame(augmented_split_features_channel)
            augmented_split_features_histogram_df = pd.DataFrame(augmented_split_features_histogram)
            augmented_split_features_coocurrence_matrix_df = pd.DataFrame(augmented_split_features_coocurrence_matrix)

            augmented_split_features = pd.merge(augmented_split_features_channel_df, augmented_split_features_histogram_df, on=['image_id', 'label'])
            augmented_split_features = pd.merge(augmented_split_features, augmented_split_features_coocurrence_matrix_df, on=['image_id', 'label'])
            augmented_split_features = pd.merge(augmented_split_features, augmented_split_features_color_df, on=['image_id', 'label'])

            augmented_split_features.loc[:, augmented_column] = [1] * len(augmented_split_features)
            split_features = pd.concat([split_features, augmented_split_features], ignore_index=True)

        split_features = self.__transform_labels(split_features, positive_class, negative_class)
        return split_features

    def transform(self, train_images: list, test_images: list, augmented_column: str, gabor_column: str) -> tuple[tuple[pd.DataFrame, pd.DataFrame], list[str]]:
        train_features = self.transform_dataset(train_images, augmented_column, gabor_column, augment=self.config.use_augmentation)
        test_features = self.transform_dataset(test_images, augmented_column, gabor_column)
        return train_features, test_features
    
    def transform_single_image(self, image) -> pd.DataFrame:
        images = [(0, image, 'test')]
        augmented_column = 'augmented'
        gabor_column = 'gabor'
        features = self.transform_dataset(images, augmented_column, gabor_column)
        features = features[features[gabor_column] == 0].drop(columns=[gabor_column, augmented_column, 'image_id', 'label'])
        return features
    
    def transform_dataset(self, images: list, augmented_column: str, gabor_column: str, augment: bool = False):
        color_features_to_extract = self.config.color_features
        channel_features_to_extract = self.config.channel_features
        histogram_features_to_extract = self.config.histogram_features
        coocurrence_matrix_features_to_extract = self.config.coocurrence_matrix_features
        correlated_features = self.config.correlated_features

        gabor_filters = gabor.create_gaborfilter()
        images_gabor = [(id, gabor.apply_filter(image, gabor_filters), label) for id, image, label in images]

        features = self.__transform_split(images, color_features_to_extract, channel_features_to_extract, histogram_features_to_extract, coocurrence_matrix_features_to_extract, self.config.to_grayscale, self.config.to_histogram, self.config.positive_class, self.config.negative_class, augmented_column, augment)
        features_gabor = self.__transform_split(images_gabor, color_features_to_extract, channel_features_to_extract, histogram_features_to_extract, coocurrence_matrix_features_to_extract, self.config.to_grayscale, self.config.to_histogram, self.config.positive_class, self.config.negative_class, augmented_column, augment)
        features.loc[:, gabor_column] = [0] * len(features)
        features_gabor.loc[:, gabor_column] = [1] * len(features_gabor)
        features = pd.concat([features, features_gabor], ignore_index=True)

        if self.config.drop_correlated_features:
            features = features.drop(columns=correlated_features)

        features = features.sort_index(axis=1)
        return features