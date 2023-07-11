import sys
import os
import shutil

def label_images_algarves(images_dir: str, labels_dir: str, new_dir: str):
    images_dirs = os.listdir(images_dir)
    labels_dirs = os.listdir(labels_dir)
    total_images_set = set(images_dirs)
    positive_images_set = set()

    for image_dir in images_dirs:
        image_name = '.'.join(image_dir.split('.')[:-1])
        for label_dir in labels_dirs:
            label_name = '.'.join(label_dir.split('.')[:-1])
            if image_name == label_name:
                positive_images_set.add(image_dir)
                break
    negative_images_set = total_images_set - positive_images_set
    return positive_images_set, negative_images_set

def create_new_dataset(
    positive_images_set: set, 
    negative_images_set: set, 
    new_dir: str, 
    positive_class: str, 
    negative_class: str,
):
    if os.path.isfile(new_dir):
        os.mkdir(new_dir)
    if os.path.isfile(os.path.join(new_dir, positive_class)):
        os.mkdir(os.path.join(new_dir, positive_class))
    if os.path.isfile(os.path.join(new_dir, negative_class)):
        os.mkdir(os.path.join(new_dir, negative_class))

    for image in positive_images_set:
        image_name = os.path.basename(image)
        shutil.copyfile(os.path.join(images_dir, image), os.path.join(new_dir, positive_class, image_name))
    for image in negative_images_set:
        image_name = os.path.basename(image)
        shutil.copyfile(os.path.join(images_dir, image), os.path.join(new_dir, negative_class, image_name))

def main(images_dir: str, labels_dir: str, new_dir: str, positive_class: str, negative_class: str):
    positive_images_set, negative_images_set = label_images_algarves(images_dir, labels_dir, new_dir)
    create_new_dataset(positive_images_set, negative_images_set, new_dir, positive_class, negative_class)

if __name__ == '__main__':
    images_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join('data', 'datasets', 'algarves', 'images')
    labels_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join('data', 'datasets', 'algarves', 'labels')
    new_dir = sys.argv[3] if len(sys.argv) > 3 else os.path.join('data', 'datasets', 'algarves', 'formatted_dataset')
    positive_class = sys.argv[4] if len(sys.argv) > 4 else 'pools'
    negative_class = sys.argv[5] if len(sys.argv) > 5 else 'no_' + positive_class
    main(images_dir, labels_dir, new_dir, positive_class, negative_class)