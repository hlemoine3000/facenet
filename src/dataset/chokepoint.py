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
import sys
import xml.etree.ElementTree as ET
from scipy import misc
import math
from src.dataset import dataset_utils

class chokepoint_data:
    def __init__(self,
                 still_path,
                 video_path,
                 pairs_path):

        self.still_path = still_path
        self.video_path = video_path
        self.pairs_path = pairs_path

        self.subject_list, self.nb_folds = self._get_subject_list()
        self.nb_subject = len(self.subject_list)
        self.nb_subject_per_fold = self.nb_subject // self.nb_folds

    # Set up for training
    def get_dataset(self,
                    fold_list,
                    video_only=False):

        assert max(fold_list) < self.nb_folds, 'Fold number {} is out of range. Maximum number of fold is {}.'.format(max(fold_list), self.nb_folds)

        dataset = []

        fold_subject_list = self._extract_fold_list(fold_list)

        for i, subject in enumerate(fold_subject_list):
            subject_video_path = os.path.join(self.video_path, subject)
            video_image_paths = facenet.get_image_paths(subject_video_path)

            if video_only:
                dataset.append(dataset_utils.ImageClass(subject, video_image_paths))
            else:
                still_images_path = os.path.join(self.still_path, 'ID' + subject + '.JPG')
                assert os.path.isfile(still_images_path), 'Still image not found at {}'.format(still_images_path)

                dataset.append(dataset_utils.COX_ImageClass(subject, video_image_paths, still_images_path))

            if not i % 100:
                print('Fetching subjects: {}/{}'.format(i, len(fold_subject_list)))

        return dataset

    # Set up for training
    def get_S2V_dataset(self,
                        fold_list,
                        video_only=False):

        assert max(
            fold_list) < self.nb_folds, 'Fold number {} is out of range. Maximum number of fold is {}.'.format(
            max(fold_list), self.nb_folds)

        dataset = []

        fold_subject_list = self._extract_fold_list(fold_list)

        for i, subject in enumerate(fold_subject_list):
            subject_video_path = os.path.join(self.video_path, subject)
            subject_still_path = os.path.join(self.still_path, subject)

            video_image_paths = dataset_utils.get_image_paths(subject_video_path)
            still_image_paths = dataset_utils.get_image_paths(subject_still_path)

            dataset.append(dataset_utils.S2V_ImageClass(subject, video_image_paths, still_image_paths))

            if not i % 100:
                print('Fetching subjects: {}/{}'.format(i, len(fold_subject_list)))

        return dataset

    # Set up for evaluation
    def get_pairs(self,
                  fold_list):
        nrof_skipped_pairs = 0
        path_list = []
        issame_list = []

        fold_subject_list = self._extract_fold_list(fold_list)

        pairs = []
        with open(self.pairs_path, 'r') as f:
            for line in f.readlines()[1:]:
                pair = line.strip().split()
                pairs.append(pair)

        for pair in pairs:
            if pair[0] in fold_subject_list:
                if len(pair) == 3:


                    path0 = facenet.add_extension(os.path.join(self.still_path, 'ID' + pair[0]))
                    path1 = facenet.add_extension(os.path.join(self.video_path, pair[0], '0000' + pair[2]))
                    issame = True

                    if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
                        path_list += (path0, path1)
                        issame_list.append(issame)
                    else:
                        nrof_skipped_pairs += 1

                elif len(pair) == 4:

                    path0 = facenet.add_extension(os.path.join(self.still_path, 'ID' + pair[0]))
                    path1 = facenet.add_extension(os.path.join(self.video_path, pair[2], '0000' + pair[3]))
                    issame = False

                    if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
                        path_list += (path0, path1)
                        issame_list.append(issame)
                    else:
                        nrof_skipped_pairs += 1

        if nrof_skipped_pairs > 0:
            print('Skipped %d image pairs' % nrof_skipped_pairs)

        return path_list, issame_list

    def get_paths_from_file(self, subject_filename, max_subject=10, max_images_per_subject=10, tag=''):
        path_list = []
        label_list = []

        subjects_list = []
        with open(subject_filename, 'r') as f:
            for line in f.readlines()[1:]:
                subjects_list.append(line.strip())

        num_subject = 0
        for subject in subjects_list:

            # Get still image
            still_image_path = os.path.join(self.still_path, subject + '_0000.JPG')
            path_list.append(still_image_path)
            label_list.append(subject + '_still' + tag)

            video_subject_dir = os.path.join(self.video_path, subject)
            subject_images_list = os.listdir(video_subject_dir)

            images_per_subject = 0
            for subject_image in subject_images_list:
                path = os.path.join(video_subject_dir, subject_image)

                if os.path.exists(path):
                    path_list.append(path)
                    label_list.append(subject + tag)
                    images_per_subject += 1

                if images_per_subject >= max_images_per_subject:
                    break

            num_subject += 1
            if num_subject >= max_subject:
                break

        return path_list, label_list

    def _extract_fold_list(self, fold_list):

        list = []
        for fold in fold_list:
            upper_idx = fold * self.nb_subject_per_fold + self.nb_subject_per_fold
            lower_idx = fold * self.nb_subject_per_fold
            list += self.subject_list[lower_idx: upper_idx]

        return list

    def _get_subject_list(self):

        subject_list = []

        with open(self.pairs_path, 'r') as f:

            nb_fold = f.readline().split('\t')[0]

            for line in f.readlines()[1:]:
                pair = line.strip().split()

                if len(pair) == 3:
                    if pair[0] not in subject_list:
                        subject_list.append(pair[0])

        return subject_list, int(nb_fold)

def calculateDistance(x1, y1, x2, y2):
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist

def setup_dataset(groundtruth_dir, video_dir, output_dir, image_size=160):

    groundtruth_file_list = [(os.path.splitext(f)[0], os.path.join(groundtruth_dir, f))
                             for f in os.listdir(groundtruth_dir)
                             if os.path.isfile(os.path.join(groundtruth_dir, f)) and
                             f.endswith(".xml")]

    num_roi = 0
    num_miss = 0

    for groundtruth_file in groundtruth_file_list:

        video_name = groundtruth_file[0]
        groundtruth_file_path = groundtruth_file[1]
        video_path = os.path.join(video_dir, groundtruth_file[0][0:6], groundtruth_file[0])

        tree = ET.parse(groundtruth_file_path)
        root = tree.getroot()

        print('\nExtracting ROI from {}.'.format(video_name))

        for child in root:

            if list(child):

                name = child.get('number')
                image_name = name + '.jpg'
                subject_id = child[0].get('id')

                subject_output_dir = os.path.join(output_dir, subject_id)
                if not os.path.exists(subject_output_dir):
                    os.makedirs(subject_output_dir)

                output_file_path = os.path.join(subject_output_dir, image_name)
                # Check if ROI already extracted
                if not os.path.isfile(output_file_path):

                    if (child[0].find('leftEye') is None) or (child[0].find('rightEye') is None):
                        num_miss += 1
                        print('\nWARNING: Annotation error at annotation {} in {}.'.format(name, groundtruth_file_path))
                    else:
                        lefteye = np.array([int(child[0][0].get('x')), int(child[0][0].get('y'))])
                        righteye = np.array([int(child[0][1].get('x')), int(child[0][1].get('y'))])

                        eye_distance = 1.5 * calculateDistance(lefteye[0], lefteye[1], righteye[0], righteye[1])
                        center = np.mean(
                            np.concatenate((np.expand_dims(lefteye, axis=0), np.expand_dims(righteye, axis=0)), axis=0),
                            axis=0)

                        image_path = os.path.join(video_path, image_name)
                        if not os.path.isfile(image_path):
                            num_miss += 1
                            print('\nWARNING: Image not found at {}'.format(image_path))
                        else:
                            img = misc.imread(image_path)
                            img_size = img.shape

                            bb = np.zeros(4, dtype=np.int32)
                            bb[0] = np.maximum(center[0] - eye_distance, 0)
                            bb[1] = np.maximum(center[1] - eye_distance, 0)
                            bb[2] = np.minimum(center[0] + eye_distance, img_size[1])
                            bb[3] = np.minimum(center[1] + eye_distance, img_size[0])
                            cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                            scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
                            misc.imsave(output_file_path, scaled)

                num_roi += 1
                if num_roi % 10 == 9:
                    print('.', end='')
                    sys.stdout.flush()

    print('')
    print('{} ROI extracted.'.format(num_roi))
    print('{} missing video sample(s).'.format(num_miss))

def combine_dataset(dataset1, dataset2):

    combined_dataset = []
    idx_combined = []

    if isinstance(dataset1[0], COX_ImageClass) and isinstance(dataset2[0], COX_ImageClass):
        # Combine dataset where still_image are same.
        for i, data1 in enumerate(dataset1):
            combined_dataset.append(data1)
            for j, data2 in enumerate(dataset2):
                if data1.name == data2.name:
                    combined_dataset[i] += data2
                    idx_combined.append(j)

        # Add sets where elements did not have been combined in dataset2.
        for i, data2 in enumerate(dataset2):
            if i not in idx_combined:
                combined_dataset.append(data2)
    elif isinstance(dataset1[0], ImageClass) and isinstance(dataset2[0], ImageClass):
        # Combine dataset where still_image are same.
        for i, data1 in enumerate(dataset1):
            combined_dataset.append(data1)
            for j, data2 in enumerate(dataset2):
                if data1.name == data2.name:
                    combined_dataset[i] += data2
                    idx_combined.append(j)

        # Add sets where elements did not have been combined in dataset2.
        for i, data2 in enumerate(dataset2):
            if i not in idx_combined:
                combined_dataset.append(data2)
    else:
        raise ValueError('Input datasets are not from the same instance or not an known instance.')

    return combined_dataset