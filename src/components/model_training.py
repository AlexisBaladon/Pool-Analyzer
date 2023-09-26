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

    def _set_data_parameters(self, train_df: pd.DataFrame, 
                             validation_df: pd.DataFrame, 
                             test_df: pd.DataFrame,
                             gabor_column: str,
                             use_gabor: bool,
                             augmented_column: str,
                             feature_columns: list[str],
                             target_column: str):
        """
        Filters out rows that don't match the current augmentation/gabor
        """
        current_train_df = train_df[train_df[gabor_column] == use_gabor]
        current_validation_df = validation_df[validation_df[gabor_column] == use_gabor]
        current_test_df = test_df[test_df[gabor_column] == use_gabor]

        to_drop_columns = [gabor_column, augmented_column]
        current_train_df = current_train_df.drop(columns=to_drop_columns)
        current_validation_df = current_validation_df.drop(columns=to_drop_columns)
        current_test_df = current_test_df.drop(columns=to_drop_columns)

        X_train = current_train_df[feature_columns]
        y_train = current_train_df[target_column]
        X_val = current_validation_df[feature_columns]
        y_val = current_validation_df[target_column]
        X_test = current_test_df[feature_columns]
        y_test = current_test_df[target_column]

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
        
    def _build_pipeline(self, feature_selection_score_function: Callable,
                        model: Model):
        pipe = Pipeline([('scaler', MinMaxScaler()),
                         ('selector', SelectKBest(feature_selection_score_function)),
                         ('model', model.model)])
        return pipe
    
    def _build_grid(self, model: Model):
        model_grid = {'model__' + key: value for key, value in 
                      model.model_parameter_grid.items()}
        select_k_grid = {'selector__k': self.config.k_features_grid.copy()}
        param_grid = {**model_grid, **select_k_grid}
        return param_grid
    
    def _update_results(self, model: Model, 
                        pred_test: list, y_test: list, 
                        train_score: float, val_score: float, 
                        start_time: float, end_time: float,
                        saved_params: dict, 
                        results: dict[str, list]):
        TP, TN, FP, FN = metrics.calculate_results(pred_test, y_test)
        classif_report = metrics.classification_report(y_test, pred_test, 
                                                       output_dict=True)
        total_samples = metrics.calculate_total_samples(TP, TN, FP, FN)
        total_time = end_time - start_time

        test_weighted_recall_score = classif_report['weighted avg']['recall']
        test_macro_recall_score = classif_report['macro avg']['recall']
        test_weighted_precision_score = classif_report['macro avg']['precision']
        test_macro_precision_score = classif_report['weighted avg']['precision']
        test_accuracy_score = classif_report['accuracy']
        test_weighted_f1_score = classif_report['weighted avg']['f1-score']
        test_macro_f1_score = classif_report['macro avg']['f1-score']
        test_positive_f1_score = classif_report['1']['f1-score']
        test_negative_f1_score = classif_report['0']['f1-score']

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

    def train(self, train_df: pd.DataFrame, 
              validation_df: pd.DataFrame,
              test_df: pd.DataFrame,
              augmented_column: str, 
              gabor_column: str) -> tuple:
        config = self.config
        results = defaultdict(list)
        best_model = None
        best_score = -1
        
        feature_selection_score_function = config.feature_selection_score_function
        non_feature_columns = [config.id_column, config.target_column, 
                               augmented_column, gabor_column]
        feature_columns = [col for col in train_df.columns 
                           if col not in non_feature_columns]

        for use_augmentation in tqdm(config.use_augmentation_grid, desc='Augmentation'):
            for use_gabor in tqdm(config.use_gabor_grid, desc='Gabor'):
                (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
                    self._set_data_parameters(train_df, 
                                              validation_df, 
                                              test_df,
                                              gabor_column, 
                                              use_gabor, 
                                              augmented_column,
                                              feature_columns, 
                                              config.target_column)

                for model in tqdm(config.models, desc='Models'):
                    best_current_grid_model = None
                    best_current_grid_score = -1
                    best_current_grid = None

                    pipe = self._build_pipeline(feature_selection_score_function, 
                                                model)
                    param_grid = self._build_grid(model)
                    
                    for grid in ParameterGrid(param_grid): # TODO: Parallelize
                        pipe.set_params(**grid)

                        start_time = datetime.now()
                        pipe.fit(X_train, y_train)
                        val_score = pipe.score(X_val, y_val)
                        end_time = datetime.now()
                        
                        if val_score > best_current_grid_score:
                            best_current_grid_model = pipe
                            best_current_grid_score = val_score
                            best_current_grid = grid

                        if val_score > best_score:
                            best_model = pipe
                            best_score = val_score

                    train_score = best_current_grid_model.score(X_train, y_train)
                    X_merged_train_val = pd.concat([X_train, X_val], axis=0)
                    y_merged_train_val = pd.concat([y_train, y_val], axis=0)
                    best_current_grid_model.fit(X_merged_train_val, y_merged_train_val)
                    pred_test = best_current_grid_model.predict(X_test)
                    
                    saved_params = {**best_current_grid, 
                                    **{'augmentation': use_augmentation, 
                                       'gabor': use_gabor}}
                    self._update_results(model, pred_test, y_test,
                                         train_score, val_score, 
                                         start_time, end_time, 
                                         saved_params, 
                                         results)

        return results, best_model
        