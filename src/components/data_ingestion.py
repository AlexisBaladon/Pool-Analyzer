import os
import dataclasses
from typing import Callable

@dataclasses.dataclass
class DataIngestionConfig:
    def __init__(self, train_data_path: str, test_data_path: str, load_images: Callable[[str], str]):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.load_images = load_images

    def __repr__(self):
        return f'DataIngestionConfig(train_data_path={self.train_data_path}, test_data_path={self.test_data_path}, load_images={self.load_images})'
    
    def __str__(self):
        return self.__repr__()    

class DataIngestor:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def __ingest_dataset(self, data_path: str, load_image: Callable[[str], str]) -> list:
        images = []
        for label in os.listdir(data_path):
            image_path = os.path.join(data_path, label)
            for image in os.listdir(image_path):
                curr_image_path = os.path.join(image_path, image)
                image = load_image(curr_image_path)
                images.append((curr_image_path, image, label))
        return images

    def ingest(self) -> tuple[list]:
        train_images = self.__ingest_dataset(self.config.train_data_path, self.config.load_images)
        test_images = self.__ingest_dataset(self.config.test_data_path, self.config.load_images)
        return train_images, test_images