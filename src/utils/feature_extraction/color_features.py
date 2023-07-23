import cv2
import numpy as np

def has_blue(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_l = np.array([100, 150, 0])
    hsv_h = np.array([140, 255, 255])
    return cv2.inRange(hsv, hsv_l, hsv_h).any()

feature_extractors = {
    'has_blue': lambda img: has_blue(img),
}

def create_color_features(labeled_images, color_features_to_extract):
    pixels_df = {'image_id': [], 'feature_name': [], 'feature_value': [], 'label': []}
    for image_id, image, label in labeled_images:
        for feature_name in color_features_to_extract:
            feature_value = feature_extractors[feature_name](image)
            pixels_df['feature_name'].append(feature_name)
            pixels_df['feature_value'].append(feature_value)
            pixels_df['label'].append(label)
            pixels_df['image_id'].append(image_id)
    return pixels_df