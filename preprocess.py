import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from skimage import io, img_as_float32, img_as_ubyte
from tqdm import tqdm
import random
import utils
import skimage
from mean_face_process import mean_face_masks


def wear_mask(landmarks, rect, input_path, output_size=(200, 200), output_base_path='faces'):
    # resize image and landmarks according to output_size
    [x0, y0, x1, y1] = rect
    image = io.imread(input_path)
    height, width = image.shape[0], image.shape[1]

    # crop
    crop_height, crop_width = np.abs(y1 - y0), np.abs(x1 - x0)
    x0, y0, x1, y1 = x0 - crop_width // 2, y0 - crop_height // 2, x1 + crop_width // 2, y1 + crop_height // 2
    x0, y0, x1, y1 = max(0, x0 - max(0, x1 - width)), max(0, y0 - max(0, y1 - height)), \
                     min(width, x1 + max(0, 0 - x0)), min(height, y1 + max(0, 0 - y0))
    image = image[y0:y1, x0:x1]
    landmarks = landmarks - np.array([x0, y0])

    # rescale
    image = resize(image, output_size)
    height_scale, width_scale = (y1 - y0) / output_size[0], (x1 - x0) / output_size[1]
    landmarks = landmarks / np.array([width_scale, height_scale])

    # wear mask
    mean_face_landmarks, mean_mask_image, mean_mask_mask = mean_face_masks.get()
    # projective_transform = skimage.transform.ProjectiveTransform()
    projective_transform = skimage.transform.AffineTransform()
    transform_indices = [0, 3, 4, 5, 6, 7]
    # transform_indices = list(range(9))
    projective_transform.estimate(landmarks[transform_indices], mean_face_landmarks[transform_indices])
    warp_mask = skimage.transform.warp(mean_mask_mask, projective_transform, output_shape=output_size)
    warp_mask = np.sum(warp_mask, 2)[..., np.newaxis]
    warp_mask_image = skimage.transform.warp(mean_mask_image, projective_transform, output_shape=output_size)
    # plt.imshow(warp_mask_image)
    # plt.scatter(mean_face_landmarks[:, 0], mean_face_landmarks[:, 1])
    # plt.scatter(face_landmarks[:, 0], face_landmarks[:, 1])
    # # plt.scatter(mean_face_landmarks[:, 0], mean_face_landmarks[:, 1])
    # plt.show()
    image = np.where(warp_mask, warp_mask_image, image)

    # save image
    output_path = os.path.join(output_base_path, input_path.split('/')[-1])
    image = img_as_ubyte(image)
    io.imsave(output_path, image)


def extract():
    # index_file_path = 'dataset/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt'
    index_file_path = 'dataset/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt'
    image_base_path = 'dataset/WFLW_images'
    target_point_indices = [60, 64, 68, 72, 54, 76, 82, 79, 85]
    output_base_path = 'dataset/test/wrong'
    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)
    with open(index_file_path) as f:
        lines = f.readlines()
    lines = lines[2*len(lines)//3:]
    points_all = []
    rect_all = []
    image_path_all = []
    count = 0
    for line in tqdm(lines):
        line = line.strip().split(' ')
        image_name = line[-1]
        line = [float(x) for x in line[:200]]
        points = []
        for i in target_point_indices:
            points.append([line[2 * i], line[2 * i + 1]])
        points = np.array(points)
        rect = [int(line[196]), int(line[197]), int(line[198]), int(line[199])]
        image_path = os.path.join(image_base_path, image_name)
        points_all.append(points)
        rect_all.append(rect)
        image_path_all.append(image_path)
        wear_mask(points, rect, image_path, output_base_path=output_base_path)
        count += 1
        # if count == 15:
        #     break
    return points_all, rect_all, image_path_all


def partition_test():
    right_path, wrong_path = 'dataset/test/right', 'dataset/test/wrong'
    test_path = 'dataset/test_anno.txt'
    test_lines = []

    fns = os.listdir(right_path)
    for fn in fns:
        test_lines.append('{} {}\n'.format("{}/{}".format(right_path, fn), 0))

    fns = os.listdir(wrong_path)
    for fn in fns:
        test_lines.append('{} {}\n'.format("{}/{}".format(wrong_path, fn), 1))

    random.shuffle(test_lines)

    with open(test_path, 'w') as f:
        f.writelines(test_lines)


def partition():
    right_path, wrong_path = 'dataset/right', 'dataset/wrong'
    train_path, validation_path = 'dataset/train_anno.txt', 'dataset/validation_anno.txt'
    train_lines, val_lines = [], []

    fns = os.listdir(right_path)
    fns_train, fns_val = fns[:int(len(fns)*0.7)], fns[int(len(fns)*0.7):]
    for fn in fns_train:
        train_lines.append('{} {}\n'.format("{}/{}".format(right_path, fn), 0))
    for fn in fns_val:
        val_lines.append('{} {}\n'.format("{}/{}".format(right_path, fn), 0))

    fns = os.listdir(wrong_path)
    fns_train, fns_val = fns[:int(len(fns)*0.7)], fns[int(len(fns)*0.7):]
    for fn in fns_train:
        train_lines.append('{} {}\n'.format("{}/{}".format(wrong_path, fn), 1))
    for fn in fns_val:
        val_lines.append('{} {}\n'.format("{}/{}".format(wrong_path, fn), 1))

    random.shuffle(train_lines)
    random.shuffle(val_lines)

    with open(train_path, 'w') as f:
        f.writelines(train_lines)
    with open(validation_path, 'w') as f:
        f.writelines(val_lines)


if __name__ == '__main__':
    # extract()
    # partition()
    partition_test()
