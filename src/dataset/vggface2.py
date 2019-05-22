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
from dataset.dataset_utils import ImageClass, add_extension, read_pairs, get_image_paths

class vggface2_data:
    def __init__(self,
                 train_images_path,
                 test_images_path,
                 pairs_path):



        self.train_images_path = train_images_path
        self.test_images_path = test_images_path
        self.pairs_path = pairs_path

        subject_list, nb_fold = self._get_subject_list()
        self.subject_list = subject_list
        self.nb_fold = nb_fold

    # Set up for training
    def get_dataset(self,
                    fold_list):

        dataset = []
        path_exp = os.path.expanduser(self.train_images_path)
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

    # Set up for evaluation
    def get_pairs(self):

        nrof_skipped_pairs = 0
        path_list = []
        issame_list = []

        pairs = read_pairs(self.pairs_path)

        path0 = ''
        path1 = ''
        issame = False

        for pair in pairs:

            if len(pair) == 3:
                path0 = add_extension(
                    os.path.join(self.test_images_path, pair[0], pair[1]))
                path1 = add_extension(
                    os.path.join(self.test_images_path, pair[0], pair[2]))
                issame = True
            elif len(pair) == 4:
                path0 = add_extension(
                    os.path.join(self.test_images_path, pair[0], pair[1]))
                path1 = add_extension(
                    os.path.join(self.test_images_path, pair[2], pair[3]))
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
