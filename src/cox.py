"""Helper for evaluation on the Labeled Faces in the Wild dataset
"""

# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import os
import numpy as np
from src import facenet


# import facenet

def evaluate(embeddings, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy, best_threshold = facenet.calculate_roc(thresholds, embeddings1, embeddings2,
                                                               np.asarray(actual_issame), nrof_folds=nrof_folds,
                                                               distance_metric=distance_metric,
                                                               subtract_mean=subtract_mean)

    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far, threshold_lowfar = facenet.calculate_val(thresholds, embeddings1, embeddings2,
                                                                np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds,
                                                                distance_metric=distance_metric,
                                                                subtract_mean=subtract_mean)

    tpr_lowfar, fpr_lowfar, acc_lowfar, _ = facenet.calculate_roc(np.array([threshold_lowfar]), embeddings1,
                                                                  embeddings2,
                                                                  np.asarray(actual_issame), nrof_folds=nrof_folds,
                                                                  distance_metric=distance_metric,
                                                                  subtract_mean=subtract_mean)

    return tpr, fpr, accuracy, val, val_std, far, best_threshold, threshold_lowfar, tpr_lowfar, acc_lowfar


def get_fold(pairs):
    person_list = []
    fold_list = []

    change_fold = False
    for pair in pairs:
        if len(pair) == 3:

            # Fold change
            if change_fold == True:
                fold_list.append(person_list)
                person_list = []
                change_fold = False

            if pair[0] not in person_list:
                person_list.append(pair[0])

        elif len(pair) == 4:
            change_fold = True


    fold_list.append(person_list)

    return fold_list


def get_paths(still_dir, video_dir, pairs, person_list):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []

    for pair in pairs:
        if len(pair) == 3:

            if pair[0] in person_list:
                path0 = add_extension(os.path.join(still_dir, pair[0] + '_' + '%04d' % int(pair[1])))
                path1 = add_extension(os.path.join(video_dir, pair[0], pair[0] + '_' + '%d' % int(pair[2])))
                issame = True

                if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
                    path_list += (path0, path1)
                    issame_list.append(issame)
                else:
                    nrof_skipped_pairs += 1

        elif len(pair) == 4:
            if pair[0] in person_list and pair[2] in person_list:
                path0 = add_extension(os.path.join(still_dir, pair[0] + '_' + '%04d' % int(pair[1])))
                path1 = add_extension(os.path.join(video_dir, pair[2], pair[2] + '_' + '%d' % int(pair[3])))
                issame = False

                if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
                    path_list += (path0, path1)
                    issame_list.append(issame)
                else:
                    nrof_skipped_pairs += 1

    if nrof_skipped_pairs > 0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)

    return path_list, issame_list

# def get_paths(cox_still_dir, cox_video_dir, ratio_neg=1.0, batch_size=None):
#     nrof_skipped_pairs = 0
#     path_list = []
#     issame_list = []
#
#     still_images_list = [f for f in os.listdir(cox_still_dir) if os.path.isfile(os.path.join(cox_still_dir, f))]
#
#     video_file_list = []
#     for path, subdirs, files in os.walk(cox_video_dir):
#         for name in files:
#             video_file_list.append(os.path.join(path, name))
#     num_video_sample = len(video_file_list)
#
#     # For each still images
#     for i, still_image in enumerate(still_images_list):
#         still_path = os.path.join(cox_still_dir, still_image)
#         if os.path.exists(still_path):
#             still_filename, ext = os.path.splitext(still_image)
#             still_filename = still_filename[:-5] # Remove '_0000' at the end
#
#             # Get positive pair sample
#             same_video_path = os.path.join(cox_video_dir, still_filename)
#
#             # For each image in the video sample
#             num_pos = 0
#             for f in os.listdir(same_video_path):
#                 video_image_path = os.path.join(same_video_path, f)
#                 if os.path.exists(video_image_path):  # Only add the pair if both paths exist
#                     path_list += (still_path, video_image_path)
#                     issame_list.append(True)
#                     num_pos += 1
#                 else:
#                     print("Video sample not found at: {}".format(video_image_path))
#
#             # Get negative pair sample
#             num_neg = int(ratio_neg * num_pos)
#             neg_cnt = 0
#             while neg_cnt < num_neg:
#                 idx = random.randint(0, num_video_sample - 1)
#                 video_file = video_file_list[idx]
#                 if os.path.exists(video_file) and not (still_filename in video_file):
#                     path_list += (still_path, video_file)
#                     issame_list.append(False)
#                     neg_cnt += 1
#
#         else:
#             print("Still image not found at: {}".format(still_path))
#
#         if not i % 50:
#             print('Generated pairs for {} subjects'.format(i))
#
#     #Remove element to fit the batch size
#     #(I don t really like this :/)
#     if batch_size:
#         if not len(path_list) % batch_size == 0:
#             number_to_remove = len(path_list) % batch_size
#             path_list = path_list[:-number_to_remove]
#             issame_list = issame_list[:-number_to_remove//2]
#             print('Removed {} pairs to fit batch size ({})'.format(number_to_remove//2, batch_size))
#
#     return path_list, issame_list


def add_extension(path):
    if os.path.exists(path + '.jpg'):
        return path + '.jpg'
    elif os.path.exists(path + '.JPG'):
        return path + '.JPG'
    elif os.path.exists(path + '.png'):
        return path + '.png'
    else:
        raise RuntimeError('No file "%s" with extension png or jpg.' % path)


def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)


class COX_ImageClass():
    "Stores the paths to images for a given class"

    def __init__(self, name, video_paths, still_path):
        self.name = name
        self.video_paths = video_paths
        self.still_path = still_path

    def __str__(self):
        return self.name + ', ' + str(len(self.video_paths)) + ' images'

    def __len__(self):
        return len(self.video_paths)

def get_dataset(still_path, video_path, train_list):

    dataset = []
    still_images_path_list = [os.path.join(still_path, f) for f in os.listdir(still_path) if os.path.isfile(os.path.join(still_path, f))]

    path_exp = os.path.expanduser(video_path)
    classes = [path for path in os.listdir(path_exp) \
               if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        if class_name in train_list:
            facedir = os.path.join(path_exp, class_name)
            video_image_paths = get_image_paths(facedir)

            # Find the corresponding still image
            still_images_path = None
            for i, path in enumerate(still_images_path_list):
                if class_name in path:
                    still_images_path = path
                    break

            dataset.append(COX_ImageClass(class_name, video_image_paths, still_images_path))

        if not i % 500:
            print('Fetching data: {}/{}'.format(i, nrof_classes))

    return dataset


class ImageClass():
    "Stores the paths to images for a given class"

    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)

def get_video_dataset(video_path, has_class_directories=True):
    dataset = []
    path_exp = os.path.expanduser(video_path)
    classes = [path for path in os.listdir(path_exp) \
               if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(ImageClass(class_name, image_paths))

        if not i % 500:
            print('Fetching data: {}/{}'.format(i, nrof_classes))

    return dataset

def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir,img) for img in images]
    return image_paths

def generate_issame(dataset, ratio_neg=1.0, batch_size=None):
    # For each still images

    path_list = []
    issame_list = []


    class_number = len(dataset)


    for data in dataset:

        still_filename, ext = os.path.splitext(data.still_path)
        still_filename = still_filename[:-5] # Remove '_0000' at the end

        # Get positive pair sample
        num_pos = 0
        for video_file in data.video_paths:
            path_list += (data.still_path, video_file)
            issame_list.append(True)
            num_pos += 1

        # Get negative pair sample
        num_neg = int(ratio_neg * num_pos)
        neg_cnt = 0
        while neg_cnt < num_neg:
            class_idx = random.randint(0, class_number - 1)
            sample_idx = random.randint(0, len(dataset[class_idx]) - 1)
            video_file = dataset[class_idx].video_paths[sample_idx]
            if os.path.exists(video_file) and not (still_filename in video_file):
                path_list += (data.still_path, video_file)
                issame_list.append(False)
                neg_cnt += 1

    # Remove element to fit the batch size
    # (I don t really like this :/)
    if batch_size:
        if not len(path_list) % batch_size == 0:
            number_to_remove = len(path_list) % batch_size
            path_list = path_list[:-number_to_remove]
            issame_list = issame_list[:-number_to_remove // 2]
            print('Removed {} pairs to fit batch size ({})'.format(number_to_remove // 2, batch_size))

    return path_list, issame_list


