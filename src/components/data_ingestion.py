import os
import dataclasses
from typing import Callable

from sklearn.model_selection import train_test_split

@dataclasses.dataclass
class DataIngestionConfig:
    load_images: Callable[[str], str] = dataclasses.field()
    seed: str = dataclasses.field()
    val_ratio: float = dataclasses.field(default=0.2)
    train_data_path: str = dataclasses.field(default=None)
    test_data_path: str = dataclasses.field(default=None)
    predict_data_path: str = dataclasses.field(default=None)

class DataIngestor:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def ingest_dataset(self, data_path: str, 
                       load_image: Callable[[str], str]) -> list:
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
        
        total_images = len(train_images) + len(test_images)
        val_ratio = self.config.val_ratio * total_images / total_images
        train_images, val_images = train_test_split(train_images, 
                                      test_size=val_ratio,
                                      random_state=self.config.seed)

        return train_images, val_images, test_images
    
    def ingest_predict(self) -> list:
        predict_images = self.ingest_dataset(self.config.predict_data_path, 
                                             self.config.load_images)
        return predict_images