import numpy as np
import cv2

# https://www.freedomvc.com/index.php/2021/10/16/gabor-filter-in-edge-detection/
def create_gaborfilter(num_filters: int=16, ksize: int=35, sigma: float=3.0, lambd: float=10.0, gamma: float=0.5, psi: int=0, theta: float=np.pi):
    filters = []
    for theta in np.arange(0, theta, theta / num_filters):
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_64F)
        kern /= 1.0 * kern.sum()
        filters.append(kern)
    return filters

def apply_filter(img, filters):
    newimage = np.zeros_like(img)
     
    depth = -1
    for kern in filters:
        image_filter = cv2.filter2D(img, depth, kern)
        np.maximum(newimage, image_filter, newimage)
    return newimage