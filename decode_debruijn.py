"""
Created by Bowen Hu
Modified by Kuo Wang
"""
import cv2
import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Decode De Bruijn sequence')
parser.add_argument('--image1', type=str, help='Path to the first image', default='imgs/dark01.png')
parser.add_argument('--image2', type=str, help='Path to the second image', default='imgs/pat01.png')
parser.add_argument('--result', type=str, help='Path to the result image', default='results/diff.png')

def find_similar_pixels(img, x, y, width, threshold):
    target_color = img[y, x]
    roi = img[y-width:y+width, x-width:x+width]
    color_diff = np.abs(roi - target_color).sum(axis=2)
    roi[color_diff < threshold] = [255, 255, 255]
    img[y-width:y+width, x-width:x+width] = roi
    return img

def classify_pixel_light_dark(image, region_width, local_threshold_percent, global_threshold):
    image_norm = np.linalg.norm(image, axis=-1)
    average_kernel = np.ones((region_width, region_width), np.float32) / (region_width**2)
    average = cv2.filter2D(image_norm, -1, average_kernel)
    return np.where((image_norm < average * local_threshold_percent) | (image_norm < global_threshold), 0, 1)

def classify_pixel(image, palette):
    image_hls = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    def create_mask(img_hls, lower, upper):
        return cv2.inRange(img_hls, np.array([lower, 0, 0]), np.array([upper, 255, 255])).astype(bool)
    
    masks = {
        'r': create_mask(image_hls, 0, 10) | create_mask(image_hls, 168, 180),
        'g': create_mask(image_hls, 40, 83),
        'b': create_mask(image_hls, 105, 127),
        'y': create_mask(image_hls, 10, 40),
        'c': create_mask(image_hls, 83, 105),
        'm': create_mask(image_hls, 127, 168),
    }

    color_classes = np.full(image.shape[:2], len(palette), dtype=np.uint8)

    for idx, (color, mask) in enumerate(masks.items()):
        color_classes[mask] = idx

    return color_classes

def swap_color(img1, img2):
    palette = {
        "r": [0, 0, 255],
        "g": [0, 255, 0],
        "b": [255, 0, 0],
        "y": [0, 255, 255],
        "c": [255, 255, 0],
        "m": [255, 0, 255],
    }
    diff = cv2.absdiff(img2, img1)

    pixel_class_dark = classify_pixel_light_dark(diff, 32, 0.995, 60)
    pixel_class = classify_pixel(diff, palette)

    modified_diff = np.zeros_like(diff)
    for i in tqdm(range(diff.shape[0])):
        for j in range(diff.shape[1]):
            if pixel_class_dark[i, j] == 0 or pixel_class[i, j] == len(palette):
                modified_diff[i, j] = [0, 0, 0]
            else:
                modified_diff[i, j] = palette[list(palette.keys())[pixel_class[i, j]]]
    return modified_diff

def main():
    
    # Read the images
    img1 = cv2.imread(args.image1)
    img2 = cv2.imread(args.image2)
    modified_diff = swap_color(img1, img2)
    cv2.imwrite(args.result, modified_diff)

if __name__ == "__main__":
    args = parser.parse_args()
    main()
