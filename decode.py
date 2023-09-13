"""
Created on Wed Sep.6 14:15:00 2023
Author: @Kuo Wang
"""
import os
import sys
print(sys.getrecursionlimit())
sys.setrecursionlimit(10000)
print(sys.getrecursionlimit())
import cv2
import random 
import numpy as np
import argparse
import time
import logging
from tqdm import tqdm
import pickle
import imageio
from collections import defaultdict
# from .decode_debruijn import swap_color

argparser = argparse.ArgumentParser(description="Decoding the diff image")
argparser.add_argument(
    "--input",
    help="path to the dark image",
    default="results/diff.png",
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
    "-p",
    "--pattern",
    help="path to the pattern image",
    default="imgs/de_bruijn_pattern_horizontal_width_2_space_8.png",
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
                if y + yy < 0 or y + yy >= skeleton.shape[0]:
                    continue
                if x + xx >= skeleton.shape[1]:
                    return
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


def decode_pattern(pattern_file: str = args.pattern):
    # reverse the key value in platte
    palette = {
    "r": [0, 0, 255],
    "g": [0, 255, 0],
    "b": [255, 0, 0],
    "y": [0, 255, 255],
    "c": [255, 255, 0],
    "m": [255, 0, 255],
}
    palette = {tuple(v): k for k, v in palette.items()}
    pattern = cv2.imread(pattern_file)
    sequence = []
    for i in range(pattern.shape[0]):
        if tuple(pattern[i, 0]) == (0, 0, 0):
            continue
        sequence.append(palette[tuple(pattern[i, 0])])
    return sequence[::2]


def find_sequence(masked_image, lines, sequence):
    """
    find the sequence in the image
    """
    palette = {
    "r": [0, 0, 255],
    "g": [0, 255, 0],
    "b": [255, 0, 0],
    "y": [0, 255, 255],
    "c": [255, 255, 0],
    "m": [255, 0, 255],
}
    palette = {tuple(v): k for k, v in palette.items()}
    lines_decode = [[] for _ in range(len(sequence))]
    sequence_dict = defaultdict(int)
    for i in range(len(sequence) - 2):
        sequence_dict[tuple(sequence[i : i + 3])] = i

    def find_pixels_vertical(y, x):
        seqs = []
        count = 0
        for yy in range(y - 1, -1, -1):
            if tuple(masked_image[yy, x]) in palette.keys():
                seqs.append(palette[tuple(masked_image[yy, x])])
                count += 1
                if count == 2:
                    break
        ind = count
        seqs.reverse()
        seqs.append(palette[tuple(masked_image[y, x])])
        count = 0
        for yy in range(y + 1, masked_image.shape[0]):
            if tuple(masked_image[yy, x]) in palette.keys():
                seqs.append(palette[tuple(masked_image[yy, x])])
                count += 1
                if count == 2:
                    break
        decode_temp = []
        for i in range(len(seqs) - 2):
            if tuple(seqs[i : i + 3]) in sequence_dict.keys():
                decode_temp.append(sequence_dict[tuple(seqs[i : i + 3])] + ind - i)
        return decode_temp

    for line in tqdm(lines):
        # random choose 20 pixel in the line
        decode_line = []
        for y, x in random.sample(line, 20):
            decode_line.extend(find_pixels_vertical(y, x))
        # find the most frequent number in the decode_line
        if len(decode_line) == 0:
            continue
        decode_line = np.array(decode_line)
        decode_line = np.bincount(decode_line)
        decode_line = np.argmax(decode_line)
        assert decode_line < len(sequence) and decode_line >= 0, "Decode line error."
        lines_decode[decode_line].extend(line)
    return lines_decode


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
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for line in lines:
        for y, x in line:
            mask[y, x] = 255
    masked_image = cv2.bitwise_and(img, img, mask=mask)
    masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
    imageio.imwrite(f'results/skeleton_{time.strftime("%Y%m%d-%H%M%S")}.png', masked_image)
    logging.info("find sequence...")
    lines_decode = find_sequence(masked_image, lines, sequence)
    # draw the lines_decode one by one and save a gif file to show the process
    logging.info("Saving gif file...")
    gif_images = []
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for i, line in enumerate(lines_decode):
        if not line:
            print(f"line {i} is None")
            continue
        for y,x in line:
            mask[y, x] = 255
        masked_image = cv2.bitwise_and(img, img, mask=mask)
        # convert bgr to rgb
        masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
        gif_images.append(masked_image)
    imageio.mimsave(f'results/gif_skeleton_{time.strftime("%Y%m%d-%H%M")}.gif', gif_images, duration=200, loop=0)
    # sava lines_decode to a pickle file
    with open(f'results/lines_decode_{time.strftime("%Y%m%d-%H%M%S")}.pkl', 'wb') as f:
        pickle.dump(lines_decode, f)

    sys.exit(0)
