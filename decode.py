"""
Created on Wed Sep.6 14:15:00 2023
Author: @Kuo Wang
"""
import os
import sys
import cv2
import numpy as np
import argparse
import time

argparser = argparse.ArgumentParser(description='Decoding the diff image')
argparser.add_argument('-i', '--input', help='path to the input image', default='results/modified_diff_mask.png')
# use current time to name the output image
argparser.add_argument('-o', '--output', help='path to the output image',default=f'results/skeleton_{time.strftime("%Y%m%d-%H%M%S")}.png')
argparser.add_argument('-t', '--threshold', help='control the color pixels', default=20, type=int)
args = argparser.parse_args()

def skeletonize(img):
    # convert to hsv 
    img = cv2.medianBlur(img, 5)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,2]

    # Ensure it's binary
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # Get a cross shaped structuring element
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    
    # Initialize the result image
    skeleton = np.zeros(img.shape, np.uint8)

    while True:
        # Open the image
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        
        # Subtract the opened image from the original image
        temp = cv2.subtract(img, opened)
        
        # Erode the original image and refine the skeleton
        eroded = cv2.erode(img, element)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img = eroded.copy()
        
        if cv2.countNonZero(img) == 0:
            break

    dilated = cv2.dilate(skeleton, element, iterations=1)
    re_skeletonized = skeletonize_colored(dilated)
    # Apply median filter to remove noise
    return re_skeletonized

def skeletonize_colored(img):
    # Ensure it's binary
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # Get a cross shaped structuring element
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    
    # Initialize the result image
    skeleton = np.zeros(img.shape, np.uint8)

    while True:
        # Open the image
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        
        # Subtract the opened image from the original image
        temp = cv2.subtract(img, opened)
        
        # Erode the original image and refine the skeleton
        eroded = cv2.erode(img, element)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img = eroded.copy()
        
        if cv2.countNonZero(img) == 0:
            break

    return skeleton


if __name__ == '__main__':
    input_path = args.input
    output_path = args.output
    # Read the image
    img = cv2.imread(input_path)  # Load in grayscale
    assert len(img.shape) == 3 and img.shape[2] != 1, "The input image should be color."

    skeleton_image = skeletonize(img)
    
    masked_image = cv2.bitwise_and(img, img, mask=skeleton_image)
    cv2.imwrite(output_path, masked_image)
    
    sys.exit(0)
