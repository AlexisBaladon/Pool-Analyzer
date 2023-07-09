from typing import Callable
import dataclasses

from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin

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
class ModelTrainerConfig:
    def __init__(
        self, 
        k_features_grid: list[int],
        feature_selection_score_function: Callable,
        models: list[Model],
        target_column: str, 
        score_criteria: str='accuracy', 
        cv: int=5,
    ):
        self.k_features_grid = k_features_grid
        self.feature_selection_score_function = feature_selection_score_function
        self.models = models
        self.target_column = target_column
        self.score_criteria = score_criteria
        self.cv = cv

    def __repr__(self):
        return f'ModelTrainerConfig(k_features_grid={self.k_features_grid}, feature_selection_score_function={self.feature_selection_score_function}, models={self.models}, target_column={self.target_column}, score_criteria={self.score_criteria}, cv={self.cv})'
    
    def __str__(self):
        return self.__repr__()

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self, train_df: pd.DataFrame, test_df: pd.DataFrame, feature_columns: list) -> tuple:
        results = {'model_name': [], 'best_params': [], 'train_score': [], 'test_score': []}
        best_model = None
        best_score = None

        for model in tqdm(self.config.models):
            pipe = Pipeline([
                #TODO: (ALEXIS)
                # Augment dataset
                # Extract features and cache non-augmented
                ('scaler', MinMaxScaler()),
                ('selector', SelectKBest(self.config.feature_selection_score_function)),
                ('model', model.model)
            ])

            model_grid = {'model__' + key: value for key, value in model.model_parameter_grid.items()}
            select_k_grid = {'selector__k': self.config.k_features_grid.copy()}
            param_grid = {**model_grid, **select_k_grid}
            grid = GridSearchCV(
                pipe,
                param_grid=param_grid, 
                cv=self.config.cv, 
                scoring=self.config.score_criteria,
                verbose=1, 
                n_jobs=-1,
            )

            grid.fit(train_df[feature_columns], train_df[self.config.target_column])
            test_score = grid.score(test_df[feature_columns], test_df[self.config.target_column])
            if best_model is None or test_score > best_score:
                best_model = model
                best_score = test_score

            results['model_name'].append(model.model_name)
            results['best_params'].append(grid.best_params_)
            results['train_score'].append(grid.best_score_)
            results['test_score'].append(test_score)

        return results, best_model
        