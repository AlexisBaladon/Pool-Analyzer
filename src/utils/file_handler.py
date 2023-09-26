import csv
import pandas as pd
import pickle

def save_features(features: pd.DataFrame, features_save_path: str):
    features.to_csv(features_save_path, index=False)

def load_features(features_save_path: str):
    return pd.read_csv(features_save_path)

def save_results(results: dict[str, list], results_save_path: str):
    with open(results_save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(results.keys())
        writer.writerows(zip(*results.values()))

def save_result(result: dict[str, str], results_save_path: str):
    with open(results_save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(result.keys())
        writer.writerow(result.values())

def save_model(model, model_save_path: str):
    with open(model_save_path, 'wb') as f:
        pickle.dump(model, f)

def load_model(model_save_path: str):
    with open(model_save_path, 'rb') as f:
        return pickle.load(f)