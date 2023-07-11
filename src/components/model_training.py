from typing import Callable
import dataclasses
from datetime import datetime

from src.utils.metrics import metrics

from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

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
        use_gabor_grid: list[int],
        use_augmentation_grid: list[int],
        feature_selection_score_function: Callable,
        models: list[Model],
        id_column: str,
        target_column: str, 
        score_criteria: str='accuracy', 
        cv: int=5,
    ):
        self.k_features_grid = k_features_grid
        self.use_gabor_grid = use_gabor_grid
        self.use_augmentation_grid = use_augmentation_grid
        self.feature_selection_score_function = feature_selection_score_function
        self.models = models
        self.id_column = id_column
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

    def train(self, train_df: pd.DataFrame, test_df: pd.DataFrame, augmented_column: str, gabor_column: str) -> tuple:
        results = {'model_name': [], 'best_params': [], 'train_score': [], 'test_accuracy_score': [], 
                   'test_f1_score': [], 'test_recall_score': [], 'test_precision_score': [],
                   'test_TP': [], 'test_TN': [], 'test_FP': [], 'test_FN': [], 'total_samples': [], 'total_time': []}
        best_model = None
        best_model_f1 = None
        non_feature_columns = [self.config.id_column, self.config.target_column, augmented_column, gabor_column]
        feature_columns = list(filter(lambda col: col not in non_feature_columns, train_df.columns))

        for use_augmentation in tqdm(self.config.use_augmentation_grid, desc='Augmentation'):
            for use_gabor in tqdm(self.config.use_gabor_grid, desc='Gabor'):
                current_train_df = train_df.copy()
                current_test_df = test_df.copy()

                if use_augmentation == 0:
                    current_train_df = current_train_df[current_train_df[augmented_column] == use_augmentation]
                    current_test_df = current_test_df[current_test_df[augmented_column] == use_augmentation]
                current_train_df = current_train_df.drop(columns=[augmented_column])
                current_test_df = current_test_df.drop(columns=[augmented_column])

                current_train_df = current_train_df[current_train_df[gabor_column] == use_gabor]
                current_test_df = current_test_df[current_test_df[gabor_column] == use_gabor]
                current_train_df = current_train_df.drop(columns=[gabor_column])
                current_test_df = current_test_df.drop(columns=[gabor_column])

                for model in tqdm(self.config.models, desc='Models'):
                    pipe = Pipeline([
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
                        verbose=0, 
                        n_jobs=-1,
                    )

                    start_time = datetime.now()
                    grid.fit(current_train_df[feature_columns], current_train_df[self.config.target_column])
                    end_time = datetime.now()

                    prediction = grid.predict(current_test_df[feature_columns])
                    TP, TN, FP, FN = metrics.calculate_results(prediction, current_test_df[self.config.target_column].tolist())
                    test_recall_score = metrics.calculate_recall(TP, FN)
                    test_precision_score = metrics.calculate_precision(TP, FP)
                    test_f1_score = metrics.calculate_f1(test_precision_score, test_recall_score)
                    test_accuracy_score = metrics.calculate_accuracy(TP, TN, FP, FN)
                    total_samples = metrics.calculate_total_samples(TP, TN, FP, FN)
                    total_time = end_time - start_time
                    if best_model is None or test_f1_score > best_model_f1:
                        best_model = grid.best_estimator_
                        best_model_f1 = test_f1_score

                    saved_params = {**grid.best_params_, **{'augmentation': use_augmentation, 'gabor': use_gabor}}
                    results['model_name'].append(model.model_name)
                    results['best_params'].append(saved_params)
                    results['train_score'].append(grid.best_score_)
                    results['test_accuracy_score'].append(test_accuracy_score)
                    results['test_f1_score'].append(test_f1_score)
                    results['test_recall_score'].append(test_recall_score)
                    results['test_precision_score'].append(test_precision_score)
                    results['test_TP'].append(TP)
                    results['test_TN'].append(TN)
                    results['test_FP'].append(FP)
                    results['test_FN'].append(FN)
                    results['total_samples'].append(total_samples)
                    results['total_time'].append(total_time)

        return results, best_model
        