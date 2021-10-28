import os
import random

import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from skimage import io
import skimage
import utils


class MeanFaceMask:
    def __init__(self, face_landmarks, mean_mask_landmarks, output_size=(200, 200)):
        '''
        :param face_landmarks:
        :param mean_mask_landmarks:
        :param masks: list of mask_dict
        '''
        self.output_size = output_size
        self.face_landmarks = face_landmarks
        self.mean_mask_landmarks = mean_mask_landmarks
        self.masks = []
        self.images = []

    def add_mask(self, image_path, mask_path, label_path):
        # get image, mask, landmarks from original label info
        image, mask, label_info = io.imread(image_path), io.imread(mask_path), utils.load_json(label_path)
        image = image[:, :, :3]
        mask = mask[:, :, :3]
        landmarks = [None] * len(label_info['points'])
        for point_info in label_info['points']:
            point = point_info['points']
            landmarks[int(point_info['label'])] = point[0]
        landmarks = np.array(landmarks)

        # perform transform to original images and points
        affine_transform = skimage.transform.AffineTransform()
        affine_transform.estimate(self.mean_mask_landmarks, landmarks)
        warp_mask = skimage.transform.warp(mask, affine_transform, output_shape=self.output_size)
        warp_image = skimage.transform.warp(image, affine_transform, output_shape=self.output_size)
        self.masks.append(warp_mask)
        self.images.append(warp_image)

    def get(self):
        # mean_face_landmarks, mean_mask_image, mean_mask_mask
        random_index = random.randrange(0, len(self.masks))
        random_mask = self.masks[random_index]
        random_image = self.images[random_index]
        return self.face_landmarks, random_image, random_mask


def get_landmarks(output_size=(200, 200)):
    mean_face_landmarks = utils.read_points('standard_human_face.txt')
    mean_face_landmarks = mean_face_landmarks[:, :2]
    mean_face_landmarks[:, 1] = 0 - mean_face_landmarks[:, 1]
    # use eye_distance to normalize
    eys_dist = np.sqrt(np.sum(mean_face_landmarks[36] - mean_face_landmarks[45]) ** 2)
    scale = eys_dist / (output_size[1] / 4)
    mean_face_landmarks = mean_face_landmarks / scale
    # put nose on the center
    trans = np.array(output_size) / 2 - mean_face_landmarks[33]
    mean_face_landmarks = mean_face_landmarks + trans

    target_indices = [36, 39, 42, 45, 33, 48, 54, 51, 57]
    mean_face_landmarks = mean_face_landmarks[target_indices]

    # plt.xlim(0, 200)
    # plt.ylim(0, 200)
    # plt.scatter(mean_face_landmarks[:, 0], mean_face_landmarks[:, 1])
    # plt.show()

    return mean_face_landmarks

mean_mask_landmarks_ls = []
xs = [[62, 138], [66, 134], [70, 130]]
ys = [[92, 138], [110, 138], [124, 138]]
for x, y in zip(xs, ys):
    x, y = np.array([x[1], (x[1] + x[0]) / 2, x[0]]), np.array([y[0], (y[0] + y[1]) / 2, y[1]])
    coords = np.vstack([x[[0, 0, 1, 2, 2]], y[[0, 1, 2, 1, 0]]]).T
    mean_mask_landmarks_ls.append(coords)

# mean_mask_landmarks_ls = [
#     np.array([[140, 90], [140, 115], [100, 140], [60, 115], [60, 90]]),
#     np.array([[136, 104], [146, 122], [100, 140], [64, 122], [64, 104]]),
#     np.array([[132, 120], [132, 130], [100, 140], [68, 130], [68, 120]])
# ]
face_landmarks = get_landmarks()
mean_face_masks = MeanFaceMask(face_landmarks, mean_mask_landmarks_ls[2])
base_path = 'masks'
for i in range(1, 6):
    path = os.path.join(base_path, '{}_json'.format(i))
    mean_face_masks.add_mask('{}/img.png'.format(path), '{}/label.png'.format(path), '{}/{}.json'.format(base_path, i))
# face_landmarks, random_image, random_mask = mean_face_masks.get()
# plt.xlim(0, 200)
# plt.ylim(0, 200)
# plt.scatter(face_landmarks[:, 0], face_landmarks[:, 1])
# plt.imshow(random_image)
# plt.show()


if __name__ == '__main__':
    pass

