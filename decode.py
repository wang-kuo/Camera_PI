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
import logging
from tqdm import tqdm
import imageio
from collections import defaultdict

argparser = argparse.ArgumentParser(description="Decoding the diff image")
argparser.add_argument(
    "-i",
    "--input",
    help="path to the input image",
    default="results/modified_diff_mask.png",
)
# use current time to name the output image
argparser.add_argument(
    "-o",
    "--output",
    help="path to the output image",
    default=f'results/skeleton_{time.strftime("%Y%m%d-%H%M%S")}.png',
)
argparser.add_argument(
    "-t", "--threshold", help="control the color pixels", default=20, type=int
)
argparser.add_argument(
    "-p", "--pattern", help="path to the pattern image", default="imgs/de_bruijn_pattern_horizontal_width_2_space_8.png"
)
args = argparser.parse_args()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def skeletonize(img):
    # convert to hsv
    img = cv2.medianBlur(img, 5)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]

    # Ensure it's binary
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Get a cross shaped structuring element
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

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
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

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


def search_line(img: np.ndarray, skeleton: np.ndarray):
    """
    search the line in the skeleton image from left to right
    """

    def search_line_helper(skeleton: np.ndarray, line: list, x: int, y: int):
        """
        search the nearest pixel in the skeleton image in the right of (x, y) with window size 5*5, and set all
        the values in the same column to be zero in skeleton.
        """
        if x >= skeleton.shape[1] - 1:
            return
        window = skeleton[y - 2 : y + 3, x + 1 : x + 6]
        if np.sum(window) == 0:
            return
        for xx in range(1, 6):
            for yy in (0, 1, -1, 2, -2):
                if skeleton[y + yy, x + xx] == 255:
                    skeleton[y - 2 : y + 3, x + xx] = 0
                    line.append((y + yy, x + xx))
                    search_line_helper(skeleton, line, x + xx, y + yy)
        return

    logging.info("Start searching lines...")
    lines = []
    for i in tqdm(range(skeleton.shape[1])):  # from left to right search each column
        if np.sum(skeleton[:, i]) == 0:  # ignore the column with all zero
            continue
        for j in np.where(skeleton[:, i] == 255)[
            0
        ]:  # search the pixel from top to bottom
            line = []
            line.append((j, i))
            search_line_helper(skeleton, line, i, j)
            if len(line) > 20:
                lines.append(line)
    logging.info(f"Found {len(lines)} lines.")
    return lines


# check line color if there are more than 50 pixel in a line with a different color, then it is a line
def check_color(imgs, lines):
    new_lines = []
    for ind, line in enumerate(lines):
        color_dict = defaultdict(list)
        for y, x in line:
            color_dict[tuple(imgs[y, x])].append((y, x))
        # sort the color_dict by the number of pixels in the same color
        color_dict = sorted(color_dict.items(), key=lambda x: len(x[1]), reverse=True)
        main_color = color_dict[0][0]
        for color, pixels in color_dict[1:]:
            if len(pixels) > 50:
                new_lines.append(pixels)
            else:
                color_dict[0][1].extend(pixels)
                # correct the color in the imgs with main_color
                for y, x in pixels:
                    imgs[y, x] = main_color
        lines[ind] = color_dict[0][1]
    lines.extend(new_lines)
    return

def decode_pattern(pattern_file:str=args.pattern):
    palette = {
        "r": [0, 0, 255],
        "g": [0, 255, 0],
        "b": [255, 0, 0],
        "y": [0, 255, 255],
        "c": [255, 255, 0],
        "m": [255, 0, 255],
    }
    # reverse the key value in platte
    palette = {tuple(v): k for k, v in palette.items()}
    pattern = cv2.imread(pattern_file)
    sequence = []
    for i in range(pattern.shape[0]):
        if tuple(pattern[i, 0]) == (0, 0, 0):
            continue
        sequence.append(palette[tuple(pattern[i, 0])])
    return sequence[::2]
            

if __name__ == "__main__":
    sequence = decode_pattern()
    input_path = args.input
    output_path = args.output
    # Read the image
    img = cv2.imread(input_path)  # Load in grayscale
    assert len(img.shape) == 3 and img.shape[2] != 1, "The input image should be color."

    skeleton_image = skeletonize(img)
    masked_image = cv2.bitwise_and(img, img, mask=skeleton_image)
    cv2.imwrite(output_path, masked_image)
    lines = search_line(img, skeleton_image.copy())
    # draw the lines and save a gif file to show the process
    check_color(img, lines)
    gif_images = []
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for line in lines:
        for y, x in line:
            mask[y, x] = 255
        masked_image = cv2.bitwise_and(img, img, mask=mask)
        # convert bgr to rgb
        masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
        gif_images.append(masked_image.copy())
    # save a gif file to show the process and replay automatically
    logging.info("There are {} frames in the gif file.".format(len(gif_images)))
    
    # imageio.mimsave(f'results/gif_skeleton_{time.strftime("%Y%m%d-%H%M%S")}.gif', gif_images, duration=200, loop=0)

    sys.exit(0)
