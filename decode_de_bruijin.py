import cv2
import numpy as np

import os
from tqdm import tqdm

import open3d as o3d

working_dir = '/mnt/ssd4/bowen/blender/projects/photo/camera_config_11_Sep_23/debruijn_rabbit_01'

palette = {
    "r": [0,0,255],
    "g": [0,255,0],
    "b": [255,0,0],
    "y": [0,255,255],
    "c": [255,255,0],
    "m": [255,0,255] }
unknown_class = len(palette)

# Read the two images
img1 = cv2.imread(os.path.join(working_dir, '01.png'))
img2 = cv2.imread(os.path.join(working_dir, '02.png'))

# Calculate the absolute difference between the two images
diff = cv2.absdiff(img2, img1)

cv2.imwrite(os.path.join(working_dir, 'diff.png'), diff)

def find_similar_pixels(img, x, y, width, threshold):
    # Get the color of the target pixel
    target_color = img[y, x]

    top = max(y-width, 0)
    bottom = min(y+width, img.shape[0])
    left = max(x-width, 0)
    right = min(x+width, img.shape[1])

    # Iterate over a small neighborhood of pixels around the target pixel
    for i in range(top, bottom):
        for j in range(left, right):
            # Get the color of the current pixel
            current_color = img[i, j]
            # Calculate the color difference between the target pixel and the current pixel
            color_diff = abs(int(target_color[0]) - int(current_color[0])) + \
                         abs(int(target_color[1]) - int(current_color[1])) + \
                         abs(int(target_color[2]) - int(current_color[2]))
            # If the color difference is below the threshold, mark the pixel as similar
            if color_diff < threshold:
                img[i, j] = [255, 255, 255]  # Mark the pixel as white
    return img

def classify_pixel_light_dark(image, region_width, local_threshold_percent, global_threshold):
    region_area = region_width * region_width
    region_threashold = int(region_area * local_threshold_percent) + 1

    pixel_class = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)

    image_norm = np.linalg.norm(image, axis=-1)

    # Calculate the average pixel value in the region
    average_kernel = np.ones((region_width, region_width), np.float32) / region_area
    average = cv2.filter2D(image_norm, -1, average_kernel)

    # If the pixel value is less than local_threshold_percent of the average, mark it as dark
    # Else mark it as light
    pixel_class[image_norm < average * local_threshold_percent] = 0
    # Or if the pixel value is less than the global threshold, mark it as dark
    pixel_class[image_norm < global_threshold] = 0

    return pixel_class


def threshold_pixel(image):
    lower_threshold = 60
    upper_threshold = 255

    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_hsv = image_hsv.astype(np.uint8)
    image_v = image_hsv[:, :, -1]

    # threshold dark values
    th_value, image_th = cv2.threshold(image_v, lower_threshold, upper_threshold,  cv2.THRESH_BINARY)
    image_th_mask = image_th.astype(bool)

    # threshold local values
    image_v_thresholded = np.zeros(image_v.shape)
    image_v_thresholded[image_th_mask] = image_v[image_th_mask]
    image_thresholded = cv2.adaptiveThreshold(image_v_thresholded.astype('uint8'), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 33, -10)

    image_thresholded[np.logical_not(image_th_mask)] = 0

    return image_thresholded


# Use local palette
# def classify_pixel(image, region_width):
#     unknown_threshold = 60

#     # Find the closest color in the palette for each pixel, and make it as a local palette
#     distances = [np.linalg.norm((v - image), axis=-1) for v in palette.values()]

#     local_palette = palette.copy()
#     for c, color in enumerate(palette.keys()):
#         local_palette[color] = np.zeros(image.shape)
#         for i in range(image.shape[0]):
#             for j in range(image.shape[1]):
#                 # find local minimal distance
#                 local_image = image[
#                     max(i-region_width, 0):min(i+region_width, image.shape[0]),
#                     max(j-region_width, 0):min(j+region_width, image.shape[1]),
#                     :]
#                 local_distances = distances[c][
#                     max(i-region_width, 0):min(i+region_width, image.shape[0]),
#                     max(j-region_width, 0):min(j+region_width, image.shape[1])]
#                 local_cloest_color = local_image[np.unravel_index(local_distances.argmin(), local_distances.shape)]
#                 local_palette[color][i, j] = local_cloest_color
    
#     distances_with_local_palette = [np.linalg.norm((v - image), axis=-1) for v in local_palette.values()]
#     color_classes = np.argmin(np.array(distances_with_local_palette).reshape(len(palette),-1), axis=0)
#     color_classes = color_classes.reshape(image.shape[0], image.shape[1])
#     # Mark the pixel as black if under the threshold
#     color_classes[np.linalg.norm(image, axis=-1) < unknown_threshold] = unknown_class

#     return color_classes


# # with region average correction
# def classify_pixel(image, region_width):
#     unknown_threshold = 60

#     region_area = region_width * region_width
#     image_norm = np.linalg.norm(image, axis=-1)
#     # Calculate the average pixel value in the region
#     average_kernel = np.ones((region_width, region_width), np.float32) / region_area
#     average = cv2.filter2D(image_norm, -1, average_kernel)
#     maxpool = skimage.measure.block_reduce(image_norm, (region_width,region_width), np.max)
#     maxpool = cv2.resize(maxpool, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

#     factor = average / maxpool

#     distances = [np.linalg.norm((np.array(v).reshape((1,1,3)).repeat(image.shape[0], axis=0).repeat(image.shape[1], axis=1) * factor.reshape(image.shape[0], image.shape[1], 1).repeat(3, axis=-1) - image), axis=-1) for v in palette.values()] 
#     color_classes = np.argmin(np.array(distances).reshape(len(palette),-1), axis=0)
#     color_classes = color_classes.reshape(image.shape[0], image.shape[1])
#     # Mark the pixel as black if under the threshold
#     color_classes[np.linalg.norm(image, axis=-1) < unknown_threshold] = unknown_class

#     return color_classes


# def classify_pixel(image):
#     unknown_threshold = 60

#     distances = [np.linalg.norm((v - image), axis=-1) for v in palette.values()] 
#     color_classes = np.argmin(np.array(distances).reshape(len(palette),-1), axis=0)
#     color_classes = color_classes.reshape(image.shape[0], image.shape[1])
#     # Mark the pixel as black if under the threshold
#     color_classes[np.linalg.norm(image, axis=-1) < unknown_threshold] = unknown_class

#     return color_classes


# use hls to classify pixel
def classify_pixel(image):
    image_hls = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_classes = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * len(palette)

    def color_mask(image_hls, lower_h, upper_h):
        lower_light, upper_light = 30, 255
        lower_saturation, upper_saturation = 0, 255
        lower_color, upper_color = np.array([lower_h, lower_light, lower_saturation]), np.array([upper_h, upper_light, upper_saturation])
        color_mask = cv2.inRange(image_hls, lower_color, upper_color).astype(bool)

        return color_mask

    # red color
    r_mask = color_mask(image_hls, 0, 10) | color_mask(image_hls, 168, 180)

    # green color
    g_mask = color_mask(image_hls, 40, 83)

    # blue color
    b_mask = color_mask(image_hls, 105, 127)

    # yellow color
    y_mask = color_mask(image_hls, 10, 40)

    # cyan color
    c_mask = color_mask(image_hls, 83, 105)

    # magenta color
    m_mask = color_mask(image_hls, 127, 168)

    color_classes[r_mask] = 0
    color_classes[g_mask] = 1
    color_classes[b_mask] = 2
    color_classes[y_mask] = 3
    color_classes[c_mask] = 4
    color_classes[m_mask] = 5

    return color_classes


def decode_skel(skel, pixel_class, pattern_sequence):
    skel = skel.astype(bool)
    select_portion = 0.9                            # portion of pixels randomly selected to decode
    search_length = 45 * 4                          # length (up or down) of pixels searched for each selected pixel
    search_depth = 4                                # depth of pixels searched for each selected pixel, 2 means 2 for up and 2 for down, totally 5 including itself
    min_interval = 5                                # minimum interval between two detected pixels
    match_threshold = search_depth + 1              # threshold of matching the pattern sequence
    total_pos_num = np.sum(skel)

    # select a portion of pixels from the skeleton to decode
    select_pos_num = int(total_pos_num * select_portion)
    true_indices = np.argwhere(skel.flatten()).flatten()          # flattened indices of nonzero pixels
    selected_indices = np.random.choice(true_indices, select_pos_num, replace=False)
    selected_skel = np.zeros_like(skel, dtype=bool)
    selected_skel.flat[selected_indices] = True                   # this is a 2-D mask

    ################ debug
    # debug_target = (1876, 2400)
    # selected_skel[debug_target] = True
    # print(f"selected_skel[debug_target]: selected_skel[{debug_target}]")
    ################

    # search upwords
    # shift the entire image to the times of search_length padding with 0
    down_shifted_skels = np.zeros((skel.shape[0], skel.shape[1], search_length), dtype=bool)
    for i in range(search_length):
        shift_length = i + 1
        shifted_skel = np.roll(skel, shift_length, axis=0)
        shifted_skel[:shift_length, :] = 0
        down_shifted_skels[:,:,i] = shifted_skel

    # find the nonzero pixel in each position
    zero_head = np.zeros(selected_skel.shape + (1,), dtype=bool)
    nonzero_indices = np.argwhere(np.concatenate((zero_head, down_shifted_skels), axis=-1)[selected_skel])
    assert len(nonzero_indices.shape) == 2
    # here detected_list is a list, each element is a array of (index, detected length for each selected pixel)
    # the index should be corresponding to the index of selected_skel.flatten()
    detected_list_u = np.split(nonzero_indices[:,:], np.unique(nonzero_indices[:, 0], return_index=True)[1][1:])
    
    selected_skel_searched_len_u_flat_sel = np.zeros(selected_skel[selected_skel].shape + (search_depth,), dtype=int)
    for detected_u_i in detected_list_u:
        current_len = 0
        #detected_u_i = detected_list_u[i]
        for d in range(search_depth):
            select_i_flat = detected_u_i[0, 0]
            assert select_i_flat == detected_u_i[-1, 0]
            target_len_min = current_len + min_interval
            target_idx = np.searchsorted(detected_u_i[:, 1], target_len_min, side='right')
            if target_idx >= len(detected_u_i):
                break
            else:
                current_len = detected_u_i[target_idx, 1]
                selected_skel_searched_len_u_flat_sel[select_i_flat, d] = current_len
    selected_skel_searched_len_u_flat = np.zeros(selected_skel.flatten().shape + (search_depth,), dtype=int)
    selected_skel_searched_len_u_flat[selected_skel.flatten()] = selected_skel_searched_len_u_flat_sel

    # search downwords
    # shift the entire image to the times of search_length padding with 0
    up_shifted_skels = np.zeros((skel.shape[0], skel.shape[1], search_length), dtype=bool)
    for i in range(search_length):
        shift_length = i + 1
        shifted_skel = np.roll(skel, -shift_length, axis=0)
        shifted_skel[-shift_length:, :] = 0
        up_shifted_skels[:,:,i] = shifted_skel
    
    # find the nonzero pixel in each position
    # reuse zero_head
    nonzero_indices = np.argwhere(np.concatenate((zero_head, up_shifted_skels), axis=-1)[selected_skel])
    assert len(nonzero_indices.shape) == 2
    # here detected_list is a list, each element is a array of (index, detected length for each selected pixel)
    # the index should be corresponding to the index of selected_skel.flatten()
    detected_list_d = np.split(nonzero_indices, np.unique(nonzero_indices[:, 0], return_index=True)[1][1:])

    selected_skel_searched_len_d_flat_sel = np.zeros(selected_skel[selected_skel].shape + (search_depth,), dtype=int)
    for i in range(len(detected_list_d)):
        current_len = 0
        detected_d_i = detected_list_d[i]
        for d in range(search_depth):
            select_i_flat = detected_d_i[0, 0]
            assert select_i_flat == detected_d_i[-1, 0]
            target_len_min = current_len + min_interval
            target_idx = np.searchsorted(detected_d_i[:, 1], target_len_min, side='right')
            if target_idx >= len(detected_d_i):
                break
            else:
                current_len = detected_d_i[target_idx, 1]
                selected_skel_searched_len_d_flat_sel[select_i_flat, d] = current_len
    selected_skel_searched_len_d_flat = np.zeros(selected_skel.flatten().shape + (search_depth,), dtype=int)
    selected_skel_searched_len_d_flat[selected_skel.flatten()] = selected_skel_searched_len_d_flat_sel
    
    searched_len_flat = np.concatenate((
        - selected_skel_searched_len_u_flat[:, ::-1],               # reverse the order of upwords searched length with negative
        np.zeros(selected_skel.flatten().shape + (1,), dtype=int),  # add a zero column in the middle
        selected_skel_searched_len_d_flat), axis=-1)                # downwords searched length
    

    ################ debug
    # print(f'searched_len_flat: {searched_len_flat[np.ravel_multi_index(debug_target, selected_skel.shape)]}')
    ################

    # TODO: I need a function to do the index2coordinate conversion,
    #       turn the nonzero element indices of selected_skel into coordinate
    #       pair the coordinate with the detected length of detected_list.
    #       Then decode the color and neighborhood information from the coordinate and detected length.
    #       I neede a function to do the coordinate2color conversion.
    def index2coordinate(index, shape):
        return np.unravel_index(index, shape)
    def coordinate2color(coordinate, unknown_class_flag=False):
        return pixel_class[coordinate] if not unknown_class_flag else unknown_class
    
    # index -> coordinate -> color
    searched_color_class_flat = np.ones(pixel_class.flatten().shape + (search_depth * 2 + 1,), dtype=int) * unknown_class
    for x, y in np.argwhere(selected_skel):
        idx = np.ravel_multi_index((x,y), selected_skel.shape)
        searched_len_list = searched_len_flat[idx]
        assert searched_len_list.shape == (search_depth * 2 + 1,)
        searched_len_list_u = searched_len_list[:search_depth]
        searched_len_list_d = searched_len_list[search_depth+1:]
        searched_color_list =\
            [coordinate2color((x + len_u, y), True if len_u == 0 else False) for len_u in searched_len_list_u] +\
            [coordinate2color((x, y), False)] +\
            [coordinate2color((x + len_d, y), True if len_d == 0 else False) for len_d in searched_len_list_d]
        searched_color_class_flat[idx] = searched_color_list

        ################ debug
        # if (x,y) == debug_target:
        #     print(f'searched_len_list_u: {searched_len_list_u}')
        #     print(f'searched_len_list_d: {searched_len_list_d}')
        #     print(f'searched_color_list: {searched_color_list}')
        ################

    # get the window view of pattern_sequence
    unmatchable_class = -97    # a class that is impossible to match
    window_size = search_depth * 2 + 1
    pattern_sequence_window = np.lib.stride_tricks.sliding_window_view(pattern_sequence, window_size)
    pattern_sequence_window_pad = np.pad(pattern_sequence_window, ((search_depth,search_depth), (0,0)), mode='constant', constant_values=unmatchable_class)
    # shape of pattern_sequence_window is (len(pattern_sequence) - window_size + 1, window_size)
    # shape of pattern_sequence_window_pad is (len(pattern_sequence) + search_depth * 2, window_size)
    # shape of searched_color_class_flat is (selected_skel.shape[0] * selected_skel.shape[1], window_size)
    searched_color_class_flat_expand = np.repeat(searched_color_class_flat[:, np.newaxis, :], pattern_sequence_window_pad.shape[0], axis=1)
    match_conv = np.sum(searched_color_class_flat_expand == pattern_sequence_window_pad, axis = -1)
    decoded_success_flat = np.logical_and((np.max(match_conv, axis=-1) >= match_threshold), selected_skel.flatten())
    decoded_skel_flat = np.argmax(match_conv, axis=-1)

    ################ debug
    # print(searched_color_class_flat[np.ravel_multi_index(debug_target, selected_skel.shape)])
    # print(f'match_conv: {match_conv[np.ravel_multi_index(debug_target, selected_skel.shape)]}')
    # print(f'decoded_skel_flat: {decoded_skel_flat[np.ravel_multi_index(debug_target, selected_skel.shape)]}')
    #assert False
    ################

    return decoded_success_flat.reshape(selected_skel.shape), decoded_skel_flat.reshape(selected_skel.shape)

    #return selected_skel, searched_color_class_flat.reshape(selected_skel.shape + (search_depth * 2 + 1,))
        

def calculate_piont_cloud(selected_skel, decoded_skel, pattern_sequence_string):
    pattern_width = 2
    pattern_space = 8

    rotation = np.array([
        0.944168862341021, 0.030781504943855866, 0.3280208199780211,
        -0.07262981700856708, 0.9905784316099, 0.1161003036627063,
        -0.3213566073180447, -0.13344238375666828, 0.9375089659039928]).reshape(3,3)
    translation = np.array([-139.38215851933654, -150.50164857383265, 195.2177303074578])
    f_proj = np.array([ 1332.401845, 2364.354831 ])
    o_proj = np.array([ 554.247635, 762.994844 ])
    f_cam = np.array([ 3572.699771, 3554.303159 ])
    o_cam = np.array([ 2461.074657, 1264.313162 ])

    point_clouds = np.array([]).reshape(0,3)

    mesh_grid_x, mesh_grid_y = np.meshgrid(np.arange(selected_skel.shape[1]), np.arange(selected_skel.shape[0]))
    assert mesh_grid_x.shape == selected_skel.shape

    def proj_idx2row(idx, width, space):
        # calculate average row number
        row_number = idx * (width + space) + 0.5 * width
        return row_number

    for row_idx in range(len(pattern_sequence_string)):
        row_number = proj_idx2row(row_idx, pattern_width, pattern_space)
        u_pixel = np.array([0, row_number])          # left side of projector in pixels
        v_pixel = np.array([1920 - 1, row_number])   # right side of projector in pixels
        u_proj = np.concatenate((((u_pixel - o_proj) / f_proj), [1]))   # 2D to 3D in projector frame
        v_proj = np.concatenate((((v_pixel - o_proj) / f_proj), [1]))   # 2D to 3D in projector frame

        n_proj = np.cross(u_proj, v_proj)           # normal vector of projector plane

        n_cam  = rotation.T @ n_proj
        q = - rotation.T @ translation              # origin of projector origin in camera frame
        D = np.dot(n_cam, q)

        d_pixel = np.vstack((
            mesh_grid_x[selected_skel][decoded_skel[selected_skel] == row_idx],
            mesh_grid_y[selected_skel][decoded_skel[selected_skel] == row_idx],))
        assert d_pixel.shape[0] == 2
        d_cam = np.concatenate((
            ((d_pixel - o_cam.reshape(2,1)) / f_cam.reshape(2,1)),
            np.ones((1, d_pixel.shape[1]))) , axis=0)
        assert d_cam.shape[0] == 3                  # 2D to 3D in camera frame

        t = (D - 0) / np.dot(n_cam, d_cam)          # 0: np.dot(n_cam, [0,0,0])

        #print(len(t))
        #assert t.shape == (1, d_pixel.shape[1])
        p_cam = d_cam * t
        if len(t) != 0:
            over_flow = np.logical_or(np.abs(t) > 10000, t < 0)
            p_cam = p_cam[:, np.logical_not(over_flow).flatten()]
            point_clouds = np.concatenate((point_clouds, p_cam.T), axis=0)
    
    return point_clouds

def save_ply_file(points, filename):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd)



if __name__ == '__main__':

    pattern_sequence_string = 'yrycrymrcgrcbrcyrccrcmrmgrmbrmyrmcrmmgggbggyggcggmgbbgbygbcgbmgybgyygycgymgcbgcygccgcmgmbgmygmcgmmbbbybbcbbm'
    ##### FIX IT
    pattern_sequence_string = ''.join(map(lambda c: {'r': 'b', 'b': 'r', 'c': 'y', 'y': 'c', 'g': 'g', 'm': 'm'}[c], pattern_sequence_string))
    #####
    char2class = dict(zip(palette.keys(), range(len(palette.keys()))))
    class2char = dict(zip(range(len(palette.keys())), palette.keys()))
    pattern_sequence = np.array([char2class[c] for c in pattern_sequence_string])

    x = 1800
    y = 1300
    threshold = 20
    search_width = 5

    modified_diff = diff.copy()

    diff_thresholded = threshold_pixel(diff)
    cv2.imwrite(os.path.join(working_dir, 'diff_th.png'), diff_thresholded)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    diff_opened = cv2.morphologyEx(diff_thresholded, cv2.MORPH_OPEN, kernel, iterations=1)
    diff_closed = cv2.morphologyEx(diff_opened, cv2.MORPH_CLOSE, kernel, iterations=1)
    cv2.imwrite(os.path.join(working_dir, 'diff_th_morph.png'), diff_closed)

    diff_skel = cv2.ximgproc.thinning(diff_closed)
    cv2.imwrite(os.path.join(working_dir, 'diff_skel.png'), diff_skel)

    diff_skel_color = np.zeros_like(diff)
    diff_skel_color[diff_skel.astype(bool)] = diff[diff_skel.astype(bool)]
    diff_skel_color = np.concatenate((np.array(list(palette.values())), np.array([[0,0,0]])), axis=0)[classify_pixel(diff_skel_color)]
    cv2.imwrite(os.path.join(working_dir, 'diff_skel_color.png'), diff_skel_color)
    #diff_with_skel_color = diff.copy()
    #diff_with_skel_color[diff_skel.astype(bool)] = diff_skel_color[diff_skel.astype(bool)]
    #cv2.imwrite('diff_with_skel_color.png', diff_with_skel_color)

    pixel_class = classify_pixel(diff)
    selected_skel, decoded_skel = decode_skel(diff_skel, pixel_class, pattern_sequence)
    print(decoded_skel.shape)
    #print(decoded_skel[np.unravel_index(np.nonzero(selected_skel.flatten())[0][-1], selected_skel.shape)])

    decoded_skel_img = np.repeat(np.expand_dims(diff_skel, axis=-1), 3, axis=-1)
    print(len(np.nonzero(selected_skel.flatten())[0]))
    print(np.nonzero(selected_skel.flatten())[0].max())
    for i in np.nonzero(selected_skel.flatten())[0][:]:
        x, y = np.unravel_index(i, selected_skel.shape)
        decoded_idx = decoded_skel.flat[i]
        color = palette[class2char[pattern_sequence[decoded_idx]]]
        #print(x, y, decoded_idx, color)
        cv2.putText(decoded_skel_img, f'{decoded_idx}', (y, x), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    cv2.imwrite(os.path.join(working_dir, 'diff_skel_decoded.png'), decoded_skel_img)

    porint_cloud = calculate_piont_cloud(selected_skel, decoded_skel, pattern_sequence_string)
    save_ply_file(porint_cloud, os.path.join(working_dir, 'point_cloud.ply'))


    # pixel_class_dark = classify_pixel_light_dark(diff, 32, 0.995, 60)
    # pixel_class = classify_pixel(diff)

    # for i in tqdm(range(diff.shape[0])):
    #     for j in range(diff.shape[1]):
    #         if pixel_class_dark[i, j] == 0 or pixel_class[i, j] == 6:
    #             modified_diff[i, j] = [0, 0, 0]
    #         else:
    #             modified_diff[i, j] = list(palette.values())[pixel_class[i, j]]


    # #modified_diff = find_similar_pixels(diff, x, y, search_width, threshold)

    # # Display the modified difference image
    # cv2.imwrite('modified_diff.png', modified_diff)