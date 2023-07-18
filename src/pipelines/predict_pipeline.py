import dataclasses

from src.utils import file_handler

from ..components import data_ingestion, data_transformation, model_prediction

@dataclasses.dataclass
class PredictPipelineConfig:
    def __init__(self, model_path: str, results_save_path: str, features_save_path: str, cache_features: bool, logger):
        self.model_path = model_path
        self.results_save_path = results_save_path
        self.features_save_path = features_save_path
        self.cache_features = cache_features
        self.logger = logger

    def __repr__(self):
        return f'TrainPipelineConfig(model_path={self.model_path}, results_save_path={self.results_save_path}, logger={self.logger})'
    
    def __str__(self):
        return self.__repr__()

class PredictPipeline:
    def __init__(
        self, 
        config: PredictPipelineConfig, 
        data_ingestor: data_ingestion.DataIngestor, 
        data_transformer: data_transformation.DataTransformer, 
        model_predictor: model_prediction.ModelPredictor
    ):
        self.config = config
        self.data_ingestor = data_ingestor
        self.data_transformer = data_transformer
        self.model_predictor = model_predictor

    def predict(self):
        logger = self.config.logger
        augmented_column = 'augmented'
        gabor_column = 'gabor'

        if self.config.cache_features:
            logger.info(f'Loading features from {self.config.features_save_path}')
            features = file_handler.load_features(self.config.features_save_path)
        else:
            logger.info(f'Ingesting data with config {self.data_ingestor.config}')
            dataset = self.data_ingestor.ingest_predict()

            logger.info(f'Transforming data with {self.data_transformer.config}')
            features = self.data_transformer.transform_dataset(dataset, augmented_column, gabor_column, augment=False)

            logger.info(f'Saving features in {self.config.features_save_path}')
            file_handler.save_features(features, self.config.features_save_path)

        logger.info(f'Loading model from {self.config.model_path}')
        model = file_handler.load_model(self.config.model_path)

        logger.info(f'Predicting features with config {self.model_predictor.config}')
        results = self.model_predictor.predict(model, features, augmented_column, gabor_column)

        logger.info(f'Saving results in {self.config.results_save_path}')
        file_handler.save_result(results, self.config.results_save_path)