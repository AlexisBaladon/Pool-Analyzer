import os
import argparse 
import random
import warnings

from sklearn.feature_selection import mutual_info_classif

from src.utils.image_processing import image_handler
from src.pipelines import train_pipeline, predict_pipeline
from src.components import data_ingestion, data_transformation, model_training, model_prediction
from models.test_models import models, models_small
from src.logger import logging

def parse_args():
    parser = argparse.ArgumentParser(description='Train or predict with model')
    parser.add_argument('--train', default=False, action='store_true', help='Train model')
    parser.add_argument('--predict', default=False, action='store_true', help='Use model')
    parser.add_argument('--id_column', type=str, default='image_id', help='ID column name')
    parser.add_argument('--target_column', type=str, default='label', help='Target column name')
    parser.add_argument('--positive_class', type=str, default='pools', help='Positive class name')
    parser.add_argument('--negative_class', type=str, default='no_pools', help='Negative class name')
    parser.add_argument('--cache_features', default=False, action='store_true', help='Cache features')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    # Train pipeline arguments
    parser.add_argument('--train_images_path', type=str, default=os.path.join('data', 'train'), help='Path to train images')
    parser.add_argument('--test_images_path', type=str, default=os.path.join('data', 'validation'), help='Path to test images')
    parser.add_argument('--train_model_save_path', type=str, default=os.path.join('models', 'best_model.pkl'), help='Path to save model')
    parser.add_argument('--train_results_save_path', type=str, default=os.path.join('results', 'results.csv'), help='Path to save results')
    parser.add_argument('--train_features_save_path', type=str, default=os.path.join('data', 'train_features.csv'), help='Path to save train features')
    parser.add_argument('--test_features_save_path', type=str, default=os.path.join('data', 'test_features.csv'), help='Path to save test features')
    parser.add_argument('--score_criteria', type=str, default='accuracy', help='Score criteria to select best model')
    parser.add_argument('--cv', type=int, default=5, help='Number of cross validation folds')
    parser.add_argument('--small_grid', default=False, action='store_true', help='Use small grid for training')
    parser.add_argument('--not_drop_correlated_features', default=False, action='store_true', help='Does not drop correlated features when activated')
    parser.add_argument('--correlated_features_path', type=str, default=os.path.join('data', 'features', 'correlated_features.txt'), help='Path to save correlated features')

    # Predict pipeline arguments
    parser.add_argument('--k_features', type=int, default=50, help='Number of features to select')
    parser.add_argument('--use_gabor', type=int, default=0, help='Use gabor filter')
    parser.add_argument('--predict_data_path', type=str, default=os.path.join('data', 'datasets', 'algarves', 'fragmented_dataset'), help='Path to predict images')
    parser.add_argument('--predict_model_path', type=str, default=os.path.join('models', 'best_model.pkl'), help='Path to save model')
    parser.add_argument('--predict_features_save_path', type=str, default=os.path.join('data', 'predict_features.csv'), help='Path to save predict features')
    parser.add_argument('--predict_results_save_path', type=str, default=os.path.join('results', 'prediction_results.csv'), help='Path to save results')

    return vars(parser.parse_args())

def main(args):
    random.seed(args['seed'])
    warnings.filterwarnings('ignore')
    
    color_features = ['has_blue']
    channel_features = ['mean', 'std', 'median', 'mode', 'min', 'max', 'range', 
                        'skewness', 'kurtosis', 'entropy', 
                        'quantile_0.25', 'quantile_0.75', 'iqr']
    histogram_features = ['mean', 'std', 'median', 'mode', 'min', 'max', 
                          'range', 'skewness', 'kurtosis', 'entropy', 'R']
    coocurrence_matrix_features = ['contrast', 'dissimilarity', 
                                   'homogeneity', 'energy', 'correlation']
    
    with open(args['correlated_features_path'], 'r') as f:
        correlated_features = [t.strip() for t in f.readlines()]

    k_features_grid = [20, 30, 40, 'all']
    use_gabor_grid = [0]
    use_augmentation_grid = [0, 1]
    
    # Data Ingestion
    ingestion_config = data_ingestion.DataIngestionConfig(
        train_data_path=args['train_images_path'],
        test_data_path=args['test_images_path'],
        predict_data_path=args['predict_data_path'],
        load_images=image_handler.load_image)
    data_ingestor = data_ingestion.DataIngestor(ingestion_config)

    # Data Transformation
    transformation_config = data_transformation.DataTransformationConfig(
        color_features=color_features,
        channel_features=channel_features,
        histogram_features=histogram_features,
        coocurrence_matrix_features=coocurrence_matrix_features,
        correlated_features=correlated_features,
        drop_correlated_features=not args['not_drop_correlated_features'],
        use_augmentation=1 if args['train'] or 1 in use_augmentation_grid else 0,
        positive_class=args['positive_class'],
        negative_class=args['negative_class'],
        to_grayscale=image_handler.to_grayscale,
        to_histogram=image_handler.to_histogram)
    data_transformer = data_transformation.DataTransformer(transformation_config)

    # Model Training/Prediction
    if args['train']:
        train_models = models_small if args['small_grid'] else models

        model_config = model_training.ModelTrainerConfig(
            k_features_grid=k_features_grid,
            use_gabor_grid=use_gabor_grid,
            use_augmentation_grid=use_augmentation_grid,
            feature_selection_score_function=mutual_info_classif,
            models=train_models,
            id_column=args['id_column'],
            target_column=args['target_column'],
            score_criteria=args['score_criteria'],
            cv=args['cv'])
        model_trainer = model_training.ModelTrainer(model_config)

        pipeline_config = train_pipeline.TrainPipelineConfig(
            model_save_path=args['train_model_save_path'],
            results_save_path=args['train_results_save_path'],
            train_features_save_path=args['train_features_save_path'],
            test_features_save_path=args['test_features_save_path'],
            cache_features=args['cache_features'],
            logger=logging)
        
        pipeline = train_pipeline.TrainPipeline(
            config=pipeline_config,
            data_ingestor=data_ingestor,
            data_transformer=data_transformer,
            model_trainer=model_trainer)
        
        pipeline.train()
    elif args['predict']:
        model_config = model_prediction.ModelPredictorConfig(
            k_features=args['k_features'],
            use_gabor=args['use_gabor'],
            feature_selection_score_function=mutual_info_classif,
            id_column=args['id_column'],
            target_column=args['target_column'],
            score_criteria=args['score_criteria'],
            cv=args['cv'])
        model_predictor = model_prediction.ModelPredictor(model_config)

        pipeline_config = predict_pipeline.PredictPipelineConfig(
            model_path=args['predict_model_path'],
            results_save_path=args['predict_results_save_path'],
            features_save_path=args['predict_features_save_path'],
            cache_features=args['cache_features'],
            logger=logging)

        pipeline = predict_pipeline.PredictPipeline(
            config=pipeline_config,
            data_ingestor=data_ingestor,
            data_transformer=data_transformer,
            model_predictor=model_predictor)

        pipeline.predict()
    else:
        print('No action selected')

# python main.py --train --small_grid --cache_features
# python main.py --train --small_grid --train_images_path data/datasets/mix/train
if __name__ == '__main__':
    args = parse_args()
    main(args)