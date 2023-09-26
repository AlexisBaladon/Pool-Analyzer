from typing import Callable
import dataclasses
from datetime import datetime
from collections import defaultdict

from src.utils.metrics import metrics

from tqdm import tqdm
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator

@dataclasses.dataclass
class Model:
    model_name: str = dataclasses.field()
    model_parameter_grid: dict[list] = dataclasses.field()
    model: BaseEstimator = dataclasses.field()

@dataclasses.dataclass
class ModelTrainerConfig:
    k_features_grid: list[int] = dataclasses.field()
    use_gabor_grid: list[int] = dataclasses.field()
    use_augmentation_grid: list[int] = dataclasses.field()
    feature_selection_score_function: Callable = dataclasses.field()
    models: list[Model] = dataclasses.field()
    id_column: str = dataclasses.field()
    target_column: str = dataclasses.field() 
    score_criteria: str = dataclasses.field(default="accuracy")

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self, 
              train_df: pd.DataFrame, 
              validation_df: pd.DataFrame,
              test_df: pd.DataFrame,
              augmented_column: str, 
              gabor_column: str) -> tuple:
        results = defaultdict(list)
        best_model = None
        best_score = -1
        best_grid = None
        
        non_feature_columns = [self.config.id_column, 
                               self.config.target_column, 
                               augmented_column, 
                               gabor_column]
        feature_columns = list(filter(lambda col: 
                                      col not in non_feature_columns, 
                                                 train_df.columns))

        for use_augmentation in tqdm(self.config.use_augmentation_grid, desc='Augmentation'):
            for use_gabor in tqdm(self.config.use_gabor_grid, desc='Gabor'):

                # Filter out rows that don't match the current augmentation/gabor
                current_train_df = train_df[train_df[gabor_column] == use_gabor]
                current_validation_df = validation_df[validation_df[gabor_column] == use_gabor]
                current_test_df = test_df[test_df[gabor_column] == use_gabor]

                current_train_df = current_train_df.drop(columns=[gabor_column, augmented_column])
                current_validation_df = current_validation_df.drop(columns=[gabor_column, augmented_column])
                current_test_df = current_test_df.drop(columns=[gabor_column, augmented_column])

                X_train = current_train_df[feature_columns]
                y_train = current_train_df[self.config.target_column]
                X_val = current_validation_df[feature_columns]
                y_val = current_validation_df[self.config.target_column]
                X_test = current_test_df[feature_columns]
                y_test = current_test_df[self.config.target_column]

                for model in tqdm(self.config.models, desc='Models'):
                    best_current_grid_model = None
                    best_current_grid_score = -1

                    feature_selection_score_function = self.config.feature_selection_score_function
                    pipe = Pipeline([('scaler', MinMaxScaler()),
                                     ('selector', SelectKBest(feature_selection_score_function)),
                                     ('model', model.model)])

                    model_grid = {'model__' + key: value  for key, value in 
                                  model.model_parameter_grid.items()}
                    select_k_grid = {'selector__k': 
                                     self.config.k_features_grid.copy()}
                    param_grid = {**model_grid, **select_k_grid}
                    
                    print()
                    for grid in ParameterGrid(param_grid):
                        print(grid)
                        pipe.set_params(**grid)

                        start_time = datetime.now()
                        pipe.fit(X_train, y_train)
                        val_score = pipe.score(X_val, y_val)
                        end_time = datetime.now()
                        
                        if val_score > best_current_grid_score:
                            best_current_grid_model = pipe
                            best_current_grid_score = val_score

                        if val_score > best_score:
                            best_model = pipe
                            best_score = val_score
                            best_grid = grid

                    train_score = best_current_grid_model.score(X_train, y_train)
                    test_prediction = best_current_grid_model.predict(X_test)
                    
                    TP, TN, FP, FN = metrics.calculate_results(test_prediction, y_test)
                    classif_report = metrics.classification_report(y_test, 
                                                                   test_prediction, 
                                                                   output_dict=True)
                    test_weighted_recall_score = classif_report['weighted avg']['recall']
                    test_macro_recall_score = classif_report['macro avg']['recall']
                    test_weighted_precision_score = classif_report['macro avg']['precision']
                    test_macro_precision_score = classif_report['weighted avg']['precision']
                    test_accuracy_score = classif_report['accuracy']
                    test_weighted_f1_score = classif_report['weighted avg']['f1-score']
                    test_macro_f1_score = classif_report['macro avg']['f1-score']
                    test_positive_f1_score = classif_report['1']['f1-score']
                    test_negative_f1_score = classif_report['0']['f1-score']

                    total_samples = metrics.calculate_total_samples(TP, TN, FP, FN)
                    total_time = end_time - start_time


                    saved_params = {**best_grid, 
                                    **{'augmentation': use_augmentation, 
                                       'gabor': use_gabor}}
                    results['model_name'].append(model.model_name)
                    results['best_params'].append(saved_params)
                    results['train_score'].append(train_score)
                    results['val_score'].append(val_score)
                    results['test_accuracy_score'].append(test_accuracy_score)
                    results['test_macro_f1_score'].append(test_macro_f1_score)
                    results['test_weighted_f1_score'].append(test_weighted_f1_score)
                    results['test_macro_recall_score'].append(test_macro_recall_score)
                    results['test_weighted_recall_score'].append(test_weighted_recall_score)
                    results['test_macro_precision_score'].append(test_macro_precision_score)
                    results['test_weighted_precision_score'].append(test_weighted_precision_score)
                    results['test_positive_f1_score'].append(test_positive_f1_score)
                    results['test_negative_f1_score'].append(test_negative_f1_score)
                    results['test_TP'].append(TP)
                    results['test_TN'].append(TN)
                    results['test_FP'].append(FP)
                    results['test_FN'].append(FN)
                    results['total_samples'].append(total_samples)
                    results['total_time'].append(total_time)

        return results, best_model
        