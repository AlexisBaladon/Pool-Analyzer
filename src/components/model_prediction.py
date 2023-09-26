from typing import Callable
import dataclasses
from datetime import datetime

from src.utils.metrics import metrics

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator

@dataclasses.dataclass
class Model:
    model_name: str = dataclasses.field()
    model_parameter_grid: dict[list] = dataclasses.field()
    model: BaseEstimator = dataclasses.field()

@dataclasses.dataclass
class ModelPredictorConfig:
    k_features: int = dataclasses.field()
    use_gabor: int = dataclasses.field()
    feature_selection_score_function: Callable = dataclasses.field()
    id_column: str = dataclasses.field()
    target_column: str = dataclasses.field() 

class ModelPredictor:
    def __init__(self, config: ModelPredictorConfig):
        self.config = config

    def predict(self, 
                model: Pipeline, 
                dataset: pd.DataFrame, 
                augmented_column: str, 
                gabor_column: str) -> dict:
        non_feature_columns = [self.config.id_column, 
                               self.config.target_column, 
                               augmented_column, gabor_column]
        feature_columns = list(filter(lambda col: 
                                      col not in non_feature_columns, dataset))
        dataset = dataset[dataset[augmented_column] == 0]

        start_time = datetime.now()
        prediction = model.predict(dataset[feature_columns])
        end_time = datetime.now()

        TP, TN, FP, FN = metrics.calculate_results(prediction, 
                                                   dataset[self.config.target_column].tolist())
        classif_report = classification_report(dataset[self.config.target_column].tolist(), 
                                               prediction, output_dict=True)
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
        results = {'model_name': model.__class__.__name__,
                   'params': saved_params,
                   'accuracy': test_accuracy_score,
                   'f1_macro': test_macro_f1_score,
                   'f1_weighted': test_weighted_f1_score,
                   'recall_macro': test_macro_recall_score,
                   'recall_weighted': test_weighted_recall_score,
                   'precision_macro': test_macro_precision_score,
                   'precision_weighted': test_weighted_precision_score,
                   'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
                   'total_samples': total_samples,
                   'total_time': total_time}

        return results
        