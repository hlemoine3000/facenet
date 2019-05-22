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

import os
import numpy as np
from src import facenet
from dataset.dataset_utils import add_extension, read_pairs
import sklearn
#import facenet

class lfw_data:
    def __init__(self,
                 images_path,
                 pairs_path):



        self.images_path = images_path
        self.pairs_path = pairs_path

        self.subject_list, self.nb_folds = self._get_subject_list()
        self.nb_subject = len(self.subject_list)
        self.nb_subject_per_fold = self.nb_subject // self.nb_folds

    # Set up for training
    def get_dataset(self,
                    fold_list):

        # assert max(fold_list) < self.nb_folds, 'Fold number {} is out of range. Maximum number of fold is {}.'.format(max(fold_list), self.nb_folds)
        #
        # dataset = []
        #
        # fold_subject_list = self._extract_fold_list(fold_list)
        #
        # for i, subject in enumerate(fold_subject_list):
        #     subject_video_path = os.path.join(self.cox_video_path, subject)
        #     video_image_paths = facenet.get_image_paths(subject_video_path)
        #
        #     dataset.append(ImageClass(subject, video_image_paths))
        #
        #     if not i % 100:
        #         print('Fetching subjects: {}/{}'.format(i, len(fold_subject_list)))
        #
        # return dataset
        return None

    # Set up for evaluation
    def get_pairs(self,
                  fold_list):

        nrof_skipped_pairs = 0
        path_list = []
        issame_list = []

        fold_subject_list = self._extract_fold_list(fold_list)

        pairs = read_pairs(self.pairs_path)

        path0 = ''
        path1 = ''
        issame = False

        for pair in pairs:
            if pair[0] in fold_subject_list:

                if len(pair) == 3:
                    path0 = add_extension(os.path.join(self.images_path, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
                    path1 = add_extension(os.path.join(self.images_path, pair[0], pair[0] + '_' + '%04d' % int(pair[2])))
                    issame = True
                elif len(pair) == 4:
                    path0 = add_extension(os.path.join(self.images_path, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
                    path1 = add_extension(os.path.join(self.images_path, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))
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
            subject_dir = os.path.join(self.images_path, subject)
            subject_images_list = os.listdir(subject_dir)

            images_per_subject = 0
            for subject_image in subject_images_list:
                path = os.path.join(subject_dir, subject_image)

                if os.path.exists(path):
                    path_list.append(path)
                    label_list.append(subject + tag)
                    images_per_subject += 1

                if images_per_subject > max_images_per_subject:
                    break

            num_subject += 1
            if num_subject > max_subject:
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


def evaluate(embeddings, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy, best_threshold = facenet.calculate_roc(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)

    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far, threshold_lowfar = facenet.calculate_val(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)

    tpr_lowfar, fpr_lowfar, acc_lowfar, _ = facenet.calculate_roc(np.array([threshold_lowfar]), embeddings1, embeddings2,
                                                                   np.asarray(actual_issame), nrof_folds=nrof_folds,
                                                                   distance_metric=distance_metric,
                                                                   subtract_mean=subtract_mean)

    return tpr, fpr, accuracy, val, val_std, far, best_threshold, threshold_lowfar, tpr_lowfar, acc_lowfar

# def get_paths(lfw_dir, pairs):
#     nrof_skipped_pairs = 0
#     path_list = []
#     issame_list = []
#     for pair in pairs:
#         if len(pair) == 3:
#             path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
#             path1 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])))
#             issame = True
#         elif len(pair) == 4:
#             path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
#             path1 = add_extension(os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))
#             issame = False
#         if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
#             path_list += (path0,path1)
#             issame_list.append(issame)
#         else:
#             nrof_skipped_pairs += 1
#     if nrof_skipped_pairs>0:
#         print('Skipped %d image pairs' % nrof_skipped_pairs)
#
#     return path_list, issame_list
#
#
# def get_paths_from_file(lfw_dir, subject_filename, max_subject=10, max_images_per_subject=10):
#     path_list = []
#     label_list = []
#
#     subjects_list = []
#     with open(subject_filename, 'r') as f:
#         for line in f.readlines()[1:]:
#             subjects_list.append(line.strip())
#
#     num_subject = 0
#     for subject in subjects_list:
#         subject_dir = os.path.join(lfw_dir, subject)
#         subject_images_list = os.listdir(subject_dir)
#
#         images_per_subject = 0
#         for subject_image in subject_images_list:
#             path = os.path.join(subject_dir, subject_image)
#
#             if os.path.exists(path):
#                 path_list.append(path)
#                 label_list.append(subject)
#                 images_per_subject += 1
#
#             if images_per_subject > max_images_per_subject:
#                 break
#
#         num_subject += 1
#         if num_subject > max_subject:
#             break
#
#
#     return path_list, label_list
