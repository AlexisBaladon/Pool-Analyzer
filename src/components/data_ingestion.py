import os
import dataclasses
from typing import Callable

@dataclasses.dataclass
class DataIngestionConfig:
    def __init__(
        self, 
        load_images: Callable[[str], str],
        train_data_path: str = None, 
        test_data_path: str = None,
        predict_data_path: str = None,
    ):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.load_images = load_images
        self.predict_data_path = predict_data_path

    def __repr__(self):
        return f'DataIngestionConfig(train_data_path={self.train_data_path}, test_data_path={self.test_data_path}, predict_data_path={self.predict_data_path}), load_images={self.load_images})'
    
    def __str__(self):
        return self.__repr__()    

class DataIngestor:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def ingest_dataset(self, data_path: str, load_image: Callable[[str], str]) -> list:
        images = []
        for label in os.listdir(data_path):
            image_path = os.path.join(data_path, label)
            for image in os.listdir(image_path):
                curr_image_path = os.path.join(image_path, image)
                image = load_image(curr_image_path)
                images.append((curr_image_path, image, label))
        return images

    def ingest_train(self) -> tuple[list]:
        train_images = self.ingest_dataset(self.config.train_data_path, 
                                           self.config.load_images)
        test_images = self.ingest_dataset(self.config.test_data_path, 
                                          self.config.load_images)
        return train_images, test_images
    
    def ingest_predict(self) -> list:
        predict_images = self.ingest_dataset(self.config.predict_data_path, 
                                             self.config.load_images)
        return predict_images