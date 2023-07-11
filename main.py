import os
import argparse 

from sklearn.feature_selection import mutual_info_classif

from src.image_processing import image_handler
from src.pipelines import train_pipeline
from src.components import data_ingestion, data_transformation, model_training
from models.test_models import models
from src.logger import logging

def parse_args():
    parser = argparse.ArgumentParser(description='Train or predict with model')
    parser.add_argument('--train', default=False, action='store_true', help='Train model')
    parser.add_argument('--predict', default=False, action='store_true', help='Use model')
    parser.add_argument('--id_column', type=str, default='image_id', help='ID column name')
    parser.add_argument('--target_column', type=str, default='label', help='Target column name')
    parser.add_argument('--positive_class', type=str, default='pools', help='Positive class name')
    parser.add_argument('--negative_class', type=str, default='no_pools', help='Negative class name')

    # Train pipeline arguments
    parser.add_argument('--train_images_path', type=str, default=os.path.join('data', 'train'), help='Path to train images')
    parser.add_argument('--test_images_path', type=str, default=os.path.join('data', 'validation'), help='Path to test images')
    parser.add_argument('--model_save_path', type=str, default=os.path.join('models', 'best_model.pkl'), help='Path to save model')
    parser.add_argument('--results_save_path', type=str, default=os.path.join('results', 'results.csv'), help='Path to save results')
    parser.add_argument('--cache_features', default=False, action='store_true', help='Cache features')
    parser.add_argument('--train_features_save_path', type=str, default=os.path.join('data', 'train_features.csv'), help='Path to save train features')
    parser.add_argument('--test_features_save_path', type=str, default=os.path.join('data', 'test_features.csv'), help='Path to save test features')
    parser.add_argument('--score_criteria', type=str, default='f1_micro', help='Score criteria to select best model')
    parser.add_argument('--cv', type=int, default=5, help='Number of cross validation folds')

    return vars(parser.parse_args())

def main(args):
    channel_features = ['mean', 'std', 'median', 'mode', 'min', 'max', 'range', 'skewness', 'kurtosis', 'entropy', 'quantile_0.25', 'quantile_0.75', 'iqr']
    histogram_features = ['mean', 'std', 'median', 'mode', 'min', 'max', 'range', 'skewness', 'kurtosis', 'entropy', 'R']
    coocurrence_matrix_features = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    k_features_grid = [40, 50, 60, 'all']
    use_gabor_grid = [0, 1]
    use_augmentation_grid = [0, 1]
    
    # Data Ingestion
    ingestion_config = data_ingestion.DataIngestionConfig(
        train_data_path=args['train_images_path'],
        test_data_path=args['test_images_path'],
        load_images=image_handler.load_image,
    )
    data_ingestor = data_ingestion.DataIngestor(ingestion_config)

    # Data Transformation
    transformation_config = data_transformation.DataTransformationConfig(
        channel_features=channel_features,
        histogram_features=histogram_features,
        coocurrence_matrix_features=coocurrence_matrix_features,
        positive_class=args['positive_class'],
        negative_class=args['negative_class'],
        to_grayscale=image_handler.to_grayscale,
        to_histogram=image_handler.to_histogram,
    )
    data_transformer = data_transformation.DataTransformer(transformation_config)

    # Model Training
    if args['train']:
        model_config = model_training.ModelTrainerConfig(
            k_features_grid=k_features_grid,
            use_gabor_grid=use_gabor_grid,
            use_augmentation_grid=use_augmentation_grid,
            feature_selection_score_function=mutual_info_classif,
            models=models,
            id_column=args['id_column'],
            target_column=args['target_column'],
            score_criteria=args['score_criteria'],
            cv=args['cv'],
        )
        model_trainer = model_training.ModelTrainer(model_config)

        pipeline_config = train_pipeline.TrainPipelineConfig(
            model_save_path=args['model_save_path'],
            results_save_path=args['results_save_path'],
            train_features_save_path=args['train_features_save_path'],
            test_features_save_path=args['test_features_save_path'],
            cache_features=args['cache_features'],
            logger=logging,
        )
        pipeline = train_pipeline.TrainPipeline(
            config=pipeline_config,
            data_ingestor=data_ingestor,
            data_transformer=data_transformer,
            model_trainer=model_trainer,
        )
        pipeline.train()
    elif args['predict']:
        print('Prediction pipeline not implemented yet')
        pass
    else:
        print('No action selected')

if __name__ == '__main__':
    args = parse_args()
    main(args)