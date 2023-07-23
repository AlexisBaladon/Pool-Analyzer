from typing import Callable
import dataclasses
from datetime import datetime
from collections import defaultdict

from src.utils.metrics import metrics

from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
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
        results = defaultdict(list)
        best_model = None
        best_model_score = None
        non_feature_columns = [self.config.id_column, self.config.target_column, augmented_column, gabor_column]
        feature_columns = list(filter(lambda col: col not in non_feature_columns, train_df.columns))

        for use_augmentation in tqdm(self.config.use_augmentation_grid, desc='Augmentation'):
            for use_gabor in tqdm(self.config.use_gabor_grid, desc='Gabor'):
                current_train_df = train_df.copy()
                current_test_df = test_df.copy()
                cv = self.config.cv

                # Filter out rows that don't match the current augmentation/gabor
                current_train_df = current_train_df[current_train_df[gabor_column] == use_gabor]
                current_test_df = current_test_df[current_test_df[gabor_column] == use_gabor]
                current_train_df = current_train_df.drop(columns=[gabor_column])
                current_test_df = current_test_df.drop(columns=[gabor_column])

                if use_augmentation == 1:
                    # Shuffle the data as done in GridSearchCV
                    shuffled_train_df = current_train_df.sample(frac=1, random_state=0)
                    train_image_indices = shuffled_train_df[self.config.id_column].tolist()
                    train_df_indices = shuffled_train_df.index.tolist()
                    n_folds = self.config.cv
                    fold_indices = [(train_image_indices[i::n_folds], train_df_indices[i::n_folds]) for i in range(n_folds)]

                    # Gridsearch cv accepts a list of (train_index, test_index) tuples
                    cv_folds = []
                    for i in range(n_folds):
                        train_indices = fold_indices[i]
                        test_indices = []
                        for j in range(n_folds):
                            if i != j:
                                test_indices.append(fold_indices[j])
                        test_image_indices, test_df_indices = [idx[0] for idx in test_indices], [idx[1] for idx in test_indices]
                        test_image_indices, test_df_indices = [item for sublist in test_image_indices for item in sublist], [item for sublist in test_df_indices for item in sublist]
                        test_indices = (test_image_indices, test_df_indices)
                        cv_folds.append((train_indices, test_indices))

                    # Augmented data should be in the same fold as the original data
                    new_cv_folds = []
                    for train_indices, test_indices in cv_folds:
                        # Find df indices of repeated images
                        current_train_image_indices, current_train_df_indices = train_indices
                        current_test_image_indices, current_test_df_indices = test_indices
                        current_repeated_test_image_indices = [index for index in current_test_image_indices if index in current_train_image_indices]
                        current_repeated_test_df_indices = [df_index for image_index, df_index in zip(current_test_image_indices, test_indices[1]) if image_index in current_repeated_test_image_indices]
                        
                        # Add image indices of test to train
                        new_current_train_indices = current_train_df_indices + current_repeated_test_df_indices
                        new_current_test_indices = [index for index in current_test_df_indices if index not in current_repeated_test_df_indices]
                        new_cv_folds.append((new_current_train_indices, new_current_test_indices))
                                    
                    cv = new_cv_folds
                    current_train_df = shuffled_train_df

                if use_augmentation == 0:
                    current_train_df = current_train_df[current_train_df[augmented_column] == use_augmentation]
                    current_test_df = current_test_df[current_test_df[augmented_column] == use_augmentation]
                current_train_df = current_train_df.drop(columns=[augmented_column])
                current_test_df = current_test_df.drop(columns=[augmented_column])
                

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
                        cv=cv, 
                        scoring=self.config.score_criteria,
                        verbose=0, 
                        n_jobs=-1,
                    )

                    start_time = datetime.now()
                    X = current_train_df[feature_columns]
                    y = current_train_df[self.config.target_column]
                    grid.fit(X, y)
                    end_time = datetime.now()

                    prediction = grid.predict(current_test_df[feature_columns])
                    TP, TN, FP, FN = metrics.calculate_results(prediction, current_test_df[self.config.target_column].tolist())
                    classif_report = classification_report(current_test_df[self.config.target_column].tolist(), prediction, output_dict=True)
                    test_weighted_recall_score = classif_report['weighted avg']['recall']
                    test_macro_recall_score = classif_report['macro avg']['recall']
                    test_weighted_precision_score = classif_report['macro avg']['precision']
                    test_macro_precision_score = classif_report['weighted avg']['precision']
                    test_accuracy_score = classif_report['accuracy']
                    test_weighted_f1_score = classif_report['weighted avg']['f1-score']
                    test_macro_f1_score = classif_report['macro avg']['f1-score']
                    test_positive_f1_score = classif_report[str(1)]['f1-score']
                    test_negative_f1_score = classif_report[str(0)]['f1-score']

                    total_samples = metrics.calculate_total_samples(TP, TN, FP, FN)
                    total_time = end_time - start_time

                    # Save only based on accuracy in the training set
                    if best_model is None or grid.best_score_ > best_model_score:
                        best_model = grid.best_estimator_
                        best_model_score = grid.best_score_

                    saved_params = {**grid.best_params_, **{'augmentation': use_augmentation, 'gabor': use_gabor}}
                    results['model_name'].append(model.model_name)
                    results['best_params'].append(saved_params)
                    results['train_score'].append(grid.best_score_)
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
        