from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from src.components.model_training import Model

SEED = 42

models = [
    Model(
        model_name='Decision Tree',
        model=DecisionTreeClassifier(),
        model_parameter_grid={
            'criterion': ['gini'],
            'max_depth': [None, 10, 20, 30],
            'random_state': [SEED],
        },
    ),
    Model(
        model_name='Random Forest',
        model=RandomForestClassifier(),
        model_parameter_grid={
            'n_estimators': [100, 200, 500],
            'criterion': ['gini'],
            'max_depth': [None, 10, 20, 30],
            'random_state': [SEED],
        },
    ),
    Model(
        model_name='KNN',
        model=KNeighborsClassifier(),
        model_parameter_grid={
            'n_neighbors': [1, 3, 5, 9, 15],
            'weights': ['distance'],
            'p': [1, 2],
            #'random_state': [SEED], KNN does not have random state
        },
    ),
    Model(
        model_name='SVM',
        model=SVC(),
        model_parameter_grid={
            'C': [0.1, 0.5, 1.0, 2.0],
            'kernel': ['poly', 'linear', 'rbf', 'sigmoid'],
            'degree': [1, 3, 5],
            'gamma': ['scale', 'auto'],
            'random_state': [SEED],
        },
    ),
]

models_small = [
    Model(
        model_name='SVM',
        model=SVC(),
        model_parameter_grid={
            'C': [2.0],
            'kernel': ['poly'],
            'degree': [5],
            'gamma': ['scale'],
            #'max_iter': [10000],
            'random_state': [SEED],
        },
    ),
]