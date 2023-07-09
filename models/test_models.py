import sys
if '..' not in sys.path:
    sys.path.append('..')

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from src.components.model_training import Model

models = [
    Model(
        model_name='Logistic Regression',
        model=LogisticRegression(),
        model_parameter_grid={
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'C': [0.1, 0.5, 1.0, 2.0, 5.0],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'max_iter': [200, 500, 1000],
            'n_jobs': [-1],
            'random_state': [42],
        },
    ),
    Model(
        model_name='Random Forest',
        model=RandomForestClassifier(),
        model_parameter_grid={
            'n_estimators': [100, 200, 500, 1000],
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth': [None, 10, 20, 50, 100],
        },
    ),
    Model(
        model_name='KNN',
        model=KNeighborsClassifier(),
        model_parameter_grid={
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'leaf_size': [10, 20, 30, 40, 50],
            'p': [1, 2],
            'n_jobs': [-1],
        },
    ),
    Model(
        model_name='SVM',
        model=SVC(),
        model_parameter_grid={
            'C': [0.1, 0.5, 1.0, 2.0, 5.0],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': [1, 2, 3, 4, 5],
            'gamma': ['scale', 'auto'],
            'max_iter': [200, 500, 1000],
            'random_state': [42],
        },
    ),
]