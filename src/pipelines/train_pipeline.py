import dataclasses
import typing

from src.utils import file_handler
from src.components import data_ingestion, data_transformation, model_training

import pandas as pd

@dataclasses.dataclass
class TrainPipelineConfig:
    model_save_path: str = dataclasses.field()
    results_save_path: str = dataclasses.field() 
    train_features_save_path: str = dataclasses.field()
    validation_features_save_path: str = dataclasses.field()
    test_features_save_path: str = dataclasses.field()
    cache_features: bool = dataclasses.field()
    logger: typing.Any = dataclasses.field()

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

    def train(self):
        logger = self.config.logger
        augmented_column = 'augmented'
        gabor_column = 'gabor'

        if self.config.cache_features:
            logger.info(f'Loading features from {self.config.train_features_save_path} and {self.config.test_features_save_path}')
            train_features = file_handler.load_features(self.config.train_features_save_path)
            validation_features = file_handler.load_features(self.config.validation_features_save_path)
            test_features = file_handler.load_features(self.config.test_features_save_path)
        else:
            logger.info(f'Ingesting data with config {self.data_ingestor.config}')
            train_images, validation_images, test_images = self.data_ingestor.ingest_train()

            logger.info(f'Transforming data with {self.data_transformer.config}')
            train_features, validation_features, test_features = self.data_transformer.transform(train_images,
                                                                                                 validation_images,
                                                                                                 test_images, 
                                                                                                 augmented_column, 
                                                                                                 gabor_column)

            train_features_save_path = self.config.train_features_save_path
            validation_features_save_path = self.config.validation_features_save_path
            test_features_save_path = self.config.test_features_save_path
            logger.info(f'Saving features in {train_features_save_path} and {test_features_save_path}')
            file_handler.save_features(train_features, train_features_save_path)
            file_handler.save_features(validation_features, validation_features_save_path)
            file_handler.save_features(test_features, test_features_save_path)

        logger.info(f'Training models with features with config {self.model_trainer.config}')
        results, best_model = self.model_trainer.train(train_features,
                                                       validation_features,
                                                       test_features, 
                                                       augmented_column, 
                                                       gabor_column)

        logger.info(f'Saving results in {self.config.results_save_path}')
        file_handler.save_results(results, self.config.results_save_path)

        logger.info(f'Saving best model in {self.config.model_save_path}')
        file_handler.save_model(best_model, self.config.model_save_path)