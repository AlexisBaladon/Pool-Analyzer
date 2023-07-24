from typing import Callable
import dataclasses
from datetime import datetime

from src.utils.metrics import metrics

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

@dataclasses.dataclass
class Model:
    def __init__(self, model_name: str, model_parameter_grid: dict[list], model):
        self.model_name = model_name
        self.model_parameter_grid = model_parameter_grid
        self.model = model

    def __repr__(self):
        return f'Model(model_name={self.model_name}, model_parameter_grid={self.model_parameter_grid}, model={self.model})'
    
    def __str__(self):
        return self.__repr__()

@dataclasses.dataclass
class ModelPredictorConfig:
    def __init__(
        self, 
        k_features: int,
        use_gabor: int,
        feature_selection_score_function: Callable,
        id_column: str,
        target_column: str, 
        score_criteria: str='f1_micro', 
        cv: int=5,
    ):
        self.k_features = k_features
        self.use_gabor = use_gabor
        self.feature_selection_score_function = feature_selection_score_function
        self.id_column = id_column
        self.target_column = target_column
        self.score_criteria = score_criteria
        self.cv = cv

    def __repr__(self):
        return f'ModelPredictorConfig(k_features={self.k_features}, feature_selection_score_function={self.feature_selection_score_function}, target_column={self.target_column}, score_criteria={self.score_criteria}, cv={self.cv})'
    
    def __str__(self):
        return self.__repr__()

class ModelPredictor:
    def __init__(self, config: ModelPredictorConfig):
        self.config = config

    def predict(self, model: Pipeline, dataset: pd.DataFrame, augmented_column: str, gabor_column: str) -> dict:
        non_feature_columns = [self.config.id_column, self.config.target_column, augmented_column, gabor_column]
        feature_columns = list(filter(lambda col: col not in non_feature_columns, dataset))
        dataset = dataset[dataset[augmented_column] == 0]

        start_time = datetime.now()
        prediction = model.predict(dataset[feature_columns])
        end_time = datetime.now()

        TP, TN, FP, FN = metrics.calculate_results(prediction, dataset[self.config.target_column].tolist())
        classif_report = classification_report(dataset[self.config.target_column].tolist(), prediction, output_dict=True)
        test_macro_f1_score = classif_report['macro avg']['f1-score']
        test_weighted_f1_score = classif_report['weighted avg']['f1-score']
        test_macro_recall_score = classif_report['macro avg']['recall']
        test_weighted_recall_score = classif_report['weighted avg']['recall']
        test_macro_precision_score = classif_report['macro avg']['precision']
        test_weighted_precision_score = classif_report['weighted avg']['precision']
        test_accuracy_score = classif_report['accuracy']
        total_samples = metrics.calculate_total_samples(TP, TN, FP, FN)
        total_time = end_time - start_time

        saved_params = {'gabor': self.config.use_gabor}
        results = {
            'model_name': model.__class__.__name__,
            'params': saved_params,
            'accuracy': test_accuracy_score,
            'f1_macro': test_macro_f1_score,
            'f1_weighted': test_weighted_f1_score,
            'recall_macro': test_macro_recall_score,
            'recall_weighted': test_weighted_recall_score,
            'precision_macro': test_macro_precision_score,
            'precision_weighted': test_weighted_precision_score,
            'TP': TP,
            'TN': TN,
            'FP': FP,
            'FN': FN,
            'total_samples': total_samples,
            'total_time': total_time,
        }

        return results
        