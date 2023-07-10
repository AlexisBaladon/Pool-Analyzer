import pandas as pd

import csv
import dataclasses
import pickle

from ..components import data_ingestion, data_transformation, model_training

@dataclasses.dataclass
class TrainPipelineConfig:
    def __init__(self, model_save_path: str, results_save_path: str, train_features_save_path: str, test_features_save_path: str, cache_features: bool, logger):
        self.model_save_path = model_save_path
        self.results_save_path = results_save_path
        self.train_features_save_path = train_features_save_path
        self.test_features_save_path = test_features_save_path
        self.cache_features = cache_features
        self.logger = logger

    def __repr__(self):
        return f'TrainPipelineConfig(model_save_path={self.model_save_path}, results_save_path={self.results_save_path}, logger={self.logger})'
    
    def __str__(self):
        return self.__repr__()

class TrainPipeline:
    def __init__(
        self, 
        config: TrainPipelineConfig, 
        data_ingestor: data_ingestion.DataIngestor, 
        data_transformer: data_transformation.DataTransformer, 
        model_trainer: model_training.ModelTrainer
    ):
        self.config = config
        self.data_ingestor = data_ingestor
        self.data_transformer = data_transformer
        self.model_trainer = model_trainer

    def __save_features(self, features: pd.DataFrame, features_save_path: str):
        features.to_csv(features_save_path, index=False)

    def __load_features(self, features_save_path: str):
        return pd.read_csv(features_save_path)

    def __save_results(self, results: dict[str, list], results_save_path: str):
        with open(results_save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(results.keys())
            writer.writerows(zip(*results.values()))

    def __save_model(self, model, model_save_path: str):
        with open(model_save_path, 'wb') as f:
            pickle.dump(model, f)

    def train(self):
        logger = self.config.logger

        if self.config.cache_features:
            logger.info(f'Loading features from {self.config.train_features_save_path} and {self.config.test_features_save_path}')
            train_features = self.__load_features(self.config.train_features_save_path)
            test_features = self.__load_features(self.config.test_features_save_path)
        else:
            logger.info(f'Ingesting data with config {self.data_ingestor.config}')
            train_images, test_images = self.data_ingestor.ingest()

            logger.info(f'Transforming data with {self.data_transformer.config}')
            train_features, test_features = self.data_transformer.transform(train_images, test_images)

            logger.info(f'Saving features in {self.config.train_features_save_path} and {self.config.test_features_save_path}')
            self.__save_features(train_features, self.config.train_features_save_path)
            self.__save_features(test_features, self.config.test_features_save_path)

        logger.info(f'Training models with features with config {self.model_trainer.config}')
        results, best_model = self.model_trainer.train(train_features, test_features)

        logger.info(f'Saving results in {self.config.results_save_path}')
        self.__save_results(results, self.config.results_save_path)

        logger.info(f'Saving best model in {self.config.model_save_path}')
        self.__save_model(best_model, self.config.model_save_path)