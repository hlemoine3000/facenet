"""Validate a face recognizer on the "Labeled Faces in the Wild" dataset (http://vis-www.cs.umass.edu/lfw/).
Embeddings are calculated using the pairs from http://vis-www.cs.umass.edu/lfw/pairs.txt and the ROC curve
is calculated and plotted. Both the model metagraph and the model parameters need to exist
in the same directory, and the metagraph should have the extension '.meta'.
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

import tensorflow as tf
import numpy as np
import argparse
import os
import sys
import csv
import re
from tensorflow.python.ops import data_flow_ops
from sklearn import metrics, preprocessing
from src import facenet
from src.dataset import cox, lfw, chokepoint
from src import config_reader

def main(args):
    config = config_reader.validation_config(args.config)

    # Extract datasets
    evaluation_list = []

    lfw_proj_paths, lfw_proj_labels = [], []
    cox_proj_paths, cox_proj_labels = [], []

    chokepoint_dataset = chokepoint.chokepoint_data('/export/livia/data/lemoineh/ChokePoint/Stills/edit_train_160',
                                                    '/export/livia/data/lemoineh/ChokePoint/train_rgb',
                                                    'chokepoint_pairs.txt')

    fold_list = [([0, 1], [2, 3, 4]),
                 ([1, 2], [3, 4, 0]),
                 ([2, 3], [4, 0, 1]),
                 ([3, 4], [0, 1, 2]),
                 ([4, 0], [1, 2, 3])]

    eval_folds = [2, 3]

    chokepoint1_paths, chokepoint1_issame = chokepoint_dataset.get_pairs(eval_folds)

    evaluation_list.append(facenet.eval_container('test', chokepoint1_paths, chokepoint1_issame))

    # Set up projection paths
    projection_paths = lfw_proj_paths + cox_proj_paths
    proj_labels = lfw_proj_labels + cox_proj_labels

    # Create label map if does not exist
    if not os.path.exists(config.emb_dir):
        os.makedirs(config.emb_dir)
    with open(os.path.join(config.emb_dir, 'meta.tsv'), "w") as meta_file:
        csvWriter = csv.writer(meta_file, delimiter='\t')
        csvWriter.writerows(np.array([proj_labels]).T)

    with tf.Graph().as_default():
      
        with tf.Session() as sess:
            
            image_paths_placeholder = tf.placeholder(tf.string, shape=(None,1), name='image_paths')
            labels_placeholder = tf.placeholder(tf.int32, shape=(None,1), name='labels')
            batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
            control_placeholder = tf.placeholder(tf.int32, shape=(None,1), name='control')
            phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
 
            nrof_preprocess_threads = 4
            eval_input_queue = data_flow_ops.FIFOQueue(capacity=2000000,
                                        dtypes=[tf.string, tf.int32, tf.int32],
                                        shapes=[(1,), (1,), (1,)],
                                        shared_name=None, name=None)
            eval_enqueue_op = eval_input_queue.enqueue_many([image_paths_placeholder, labels_placeholder, control_placeholder], name='eval_enqueue_op')
            image_batch, label_batch = facenet.create_input_pipeline(eval_input_queue, (config.image_size, config.image_size), nrof_preprocess_threads, batch_size_placeholder)
     
            # Load the model
            input_map = {'image_batch': image_batch, 'label_batch': label_batch, 'phase_train': phase_train_placeholder}
            facenet.load_model(config.model, input_map=input_map)

            # Get output tensor
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
#              
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(coord=coord, sess=sess)

            model = facenet.model_container(enqueue_op=eval_enqueue_op,
                                            embeddings=embeddings,
                                            labels_batch=label_batch,
                                            image_paths_placeholder=image_paths_placeholder,
                                            labels_placeholder=labels_placeholder,
                                            batch_size_placeholder=batch_size_placeholder,
                                            phase_train_placeholder=phase_train_placeholder,
                                            control_placeholder=control_placeholder,
                                            batch_size=config.batch_size)

            # Evaluate
            if evaluation_list:
                for eval_set in evaluation_list:
                    evaluate(config,
                             model,
                             sess,
                             eval_set.image_paths,
                             eval_set.actual_issame,
                             2,
                             tag=eval_set.name)



def evaluate(config: config_reader.validation_config,
             model: facenet.model_container,
             sess,
             image_paths,
             actual_issame,
             nrof_folds,
             tag='eval'):

    # Run forward pass to calculate embeddings
    print('Runnning forward pass on ' + tag + ' images')

    still_image_paths = []
    still_ref = []

    num_still = 80
    # still_image_dir = '/export/livia/data/lemoineh/ChokePoint/SimGan_test/Stills_160'
    # still_image_dir = '/export/livia/data/lemoineh/ChokePoint/SimGan_test/Stills_160_3'
    still_image_dir = '/export/livia/data/lemoineh/ChokePoint/3DMM/3DMM-jpg_MTCNN160'
    subject_list = [dI for dI in os.listdir(still_image_dir) if os.path.isdir(os.path.join(still_image_dir, dI))]

    for subject in subject_list:
        subject_path = os.path.join(still_image_dir, subject)
        image_name_list = [os.path.join(subject_path, f) for f in os.listdir(subject_path) if
                           os.path.isfile(os.path.join(subject_path, f))]
        still_image_paths += image_name_list[:num_still]
        still_ref.append(subject)

    # Enqueue one epoch of image paths and labels
    nrof_embeddings = len(still_image_paths)  # nrof_pairs * nrof_images_per_pair
    nrof_images = nrof_embeddings

    batch_size = num_still
    assert not batch_size == 0, 'Fitting batch size not found for {} images.'.format(nrof_images)

    labels_array = np.expand_dims(np.arange(0, nrof_images), 1)
    image_paths_array = np.expand_dims(np.repeat(np.array(still_image_paths), 1), 1)

    control_array = np.zeros_like(labels_array, np.int32)

    sess.run(model.enqueue_op,
             {model.image_paths_placeholder: image_paths_array, model.labels_placeholder: labels_array,
              model.control_placeholder: control_array})

    embedding_size = int(model.embeddings.get_shape()[1])
    assert nrof_images % batch_size == 0, 'The number of {} images ({}) must be an integer multiple of the batch size ({})'.format(
        tag, nrof_images, batch_size)
    nrof_batches = nrof_images // batch_size
    emb_array = np.zeros((nrof_images, embedding_size))
    lab_array = np.zeros((nrof_images,))
    for i in range(nrof_batches):
        feed_dict = {model.phase_train_placeholder: False, model.batch_size_placeholder: batch_size}
        emb, lab = sess.run([model.embeddings, model.labels_batch], feed_dict=feed_dict)
        lab_array[lab] = lab
        emb_array[lab, :] = emb
        if i % 10 == 9:
            print('.', end='')
            sys.stdout.flush()
    print('')
    still_embeddings = emb_array

    still_emb_dict = {}
    for i in range(len(still_ref)):
        still_emb_dict[still_ref[i].zfill(4)] = still_embeddings[i * num_still: i * num_still + num_still]
        # still_emb_list.append((still_ref[i].zfill(4), still_embeddings[i * num_still: i * num_still + num_still]))



    # Enqueue one epoch of image paths and labels
    nrof_embeddings = len(image_paths) // 2  # nrof_pairs * nrof_images_per_pair
    nrof_flips = 2 if config.use_flipped_images else 1
    nrof_images = nrof_embeddings * nrof_flips

    batch_size = 0
    if nrof_images < 100:
        batch_size = nrof_images
    else:
        # Try multiple batch size to fit number or images
        for i in range(1, 101):
            if nrof_images % i == 0:
                batch_size = i
    assert not batch_size == 0, 'Fitting batch size not found for {} images.'.format(nrof_images)
    print('Batch size: {}'.format(batch_size))

    labels_array = np.expand_dims(np.arange(0,nrof_images),1)
    image_paths_array = np.expand_dims(np.repeat(np.array(image_paths),nrof_flips),1)

    still_array = []
    still_paths = image_paths_array[0::2]
    for still in still_paths:
        still_array.append(re.search('ID(.+?).JPG', still[0]).group(1))

    image_paths_array = image_paths_array[1::2]

    control_array = np.zeros_like(labels_array, np.int32)
    if config.use_fixed_image_standardization:
        control_array += np.ones_like(labels_array)*facenet.FIXED_STANDARDIZATION
    if config.use_flipped_images:
        # Flip every second image
        control_array += (labels_array % 2)*facenet.FLIP

    sess.run(model.enqueue_op, {model.image_paths_placeholder: image_paths_array, model.labels_placeholder: labels_array, model.control_placeholder: control_array})
    
    embedding_size = int(model.embeddings.get_shape()[1])
    assert nrof_images % batch_size == 0, 'The number of {} images ({}) must be an integer multiple of the batch size ({})'.format(tag, nrof_images, batch_size)
    nrof_batches = nrof_images // batch_size
    emb_array = np.zeros((nrof_images, embedding_size))
    lab_array = np.zeros((nrof_images,))
    for i in range(nrof_batches):
        feed_dict = {model.phase_train_placeholder:False, model.batch_size_placeholder: batch_size}
        emb, lab = sess.run([model.embeddings, model.labels_batch], feed_dict=feed_dict)
        lab_array[lab] = lab
        emb_array[lab, :] = emb
        if i % 10 == 9:
            print('.', end='')
            sys.stdout.flush()
    print('')
    embeddings = np.zeros((nrof_embeddings, embedding_size*nrof_flips))
    if config.use_flipped_images:
        # Concatenate embeddings for flipped and non flipped version of the images
        embeddings[:,:embedding_size] = emb_array[0::2,:]
        embeddings[:,embedding_size:] = emb_array[1::2,:]
    else:
        embeddings = emb_array

    # embeddings,
    # still_array,
    # num_still,

    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)

    tpr = 0
    fpr = 0

    dist = np.zeros(embeddings.shape[0])
    for i, emb in enumerate(embeddings):

        subject = still_array[i]

        embeddings1 = still_emb_dict[subject]

        embeddings2 = np.zeros((num_still, embedding_size))
        for j in range(num_still):
            embeddings2[j] = emb

        all_dist = facenet.distance(embeddings1, embeddings2, config.distance_metric)
        dist[i] = np.mean(all_dist)

    dist2 = dist.reshape(-1, 1)
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(dist2)
    norm_dist = scaler.transform(dist2)

    scores = np.squeeze(np.ones(norm_dist.shape) - norm_dist)

    pAUC = metrics.roc_auc_score(actual_issame, scores, max_fpr=0.2)
    print('pAUC FAR=20%: {}'.format(pAUC))

    AP = metrics.average_precision_score(actual_issame, scores)
    print('AP: {}'.format(AP))

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str,
        help='Path to the configuration file', default='config/validation.ini')

    return parser.parse_args(argv)


if __name__ == '__main__':

    main(parse_arguments(sys.argv[1:]))
