import sys
import os
import shutil

import numpy as np
import PIL
from transformers import CLIPProcessor, CLIPModel

def fetch_dataset(dataset_dir: str):
    dataset = []
    for image in os.listdir(dataset_dir):
        image_path = os.path.join(dataset_dir, image)
        image = np.asarray(PIL.Image.open(image_path))
        dataset.append((image, image_path))
    return dataset

def fragment_dataset(dataset: list[tuple], new_image_size: tuple[int, int]) -> list[tuple]:
    segmented_dataset = []
    for (image, image_dir) in dataset:
        grid = [(x, y) for x in range(0, image.shape[0], new_image_size[0]) for y in range(0, image.shape[1], new_image_size[1])]
        # Segment image pieces only if the piece fits the image
        segmented_images = [image[x:x+new_image_size[0], y:y+new_image_size[1]] for (x, y) in grid if x+new_image_size[0] <= image.shape[0] and y+new_image_size[1] <= image.shape[1]]
        for idx, segmented_image in enumerate(segmented_images):
            new_name = f'{os.path.basename(image_dir)}_{new_image_size[0]}x{new_image_size[1]}_{idx}.png'
            segmented_dataset.append((segmented_image, new_name))
    return segmented_dataset

def label_dataset(dataset: list[tuple]) -> list[tuple]:
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    positive_images = []
    negative_images = []
    for idx, (image, image_dir) in enumerate(dataset):
        inputs = processor(text=["pool", "other"], images=image, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        if probs[0][0] > probs[0][1]:
            positive_images.append((image, image_dir))
        else:
            negative_images.append((image, image_dir))
    return positive_images, negative_images

def persist_dataset(dataset: list, dataset_dir: str):
    for (image, image_dir) in dataset:
        image_name = os.path.basename(image_dir)
        image_path = os.path.join(dataset_dir, image_name)
        PIL.Image.fromarray(image).save(image_path)

if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_fetch_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root_dir, 'data', 'datasets', 'algarves', 'formatted_dataset', 'pools')
    new_image_x = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    new_image_y = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    dataset_persist_dir = sys.argv[4] if len(sys.argv) > 4 else os.path.join(root_dir, 'data', 'datasets', 'algarves', 'fragmented_dataset')

    new_image_size = (new_image_x, new_image_y)
    dataset_classes = ('no_pools', 'pools')
    persist_dataset_dir_positive = os.path.join(dataset_persist_dir, dataset_classes[1])
    persist_dataset_dir_negative = os.path.join(dataset_persist_dir, dataset_classes[0])
 
    for new_dataset in [dataset_persist_dir, persist_dataset_dir_positive, persist_dataset_dir_negative]:
        if os.path.isdir(new_dataset):
            shutil.rmtree(new_dataset)
        os.mkdir(new_dataset)

    dataset = fetch_dataset(dataset_fetch_dir)
    segmented_dataset = fragment_dataset(dataset, new_image_size)
    positive_images, negative_images = label_dataset(segmented_dataset)
    persist_dataset(positive_images, os.path.join(dataset_persist_dir, dataset_classes[1]))
    persist_dataset(negative_images, os.path.join(dataset_persist_dir, dataset_classes[0]))