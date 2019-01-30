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
import itertools
from tensorflow.python.ops import data_flow_ops
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
from src import facenet
from src import lfw
from src import cox
from src import config_reader

def main(args):
    config = config_reader.validation_config(args.config)

    # Extract datasets
    lfw_paths, lfw_actual_issame = [], []
    cox_paths, cox_actual_issame = [], []

    lfw_proj_paths, lfw_proj_labels = [], []
    cox_proj_paths, cox_proj_labels = [], []

    # LFW
    if config.eval_lfw:

        print('Fetching LFW evaluation pairs.')
        # Read the file containing the pairs used for testing
        pairs = lfw.read_pairs(os.path.expanduser(config.lfw_pairs))
        # Get the paths for the corresponding images
        lfw_paths, lfw_actual_issame = lfw.get_paths(os.path.expanduser(config.lfw_dir), pairs)
        # Get the paths for embeddings projection

    # Get the paths for embeddings projection
    if config.save_lfw_projections:

        print('Fetching LFW projections samples.')
        lfw_proj_paths, lfw_proj_labels = lfw.get_paths_from_file(config.lfw_dir, config.lfw_projection,
                                                                      max_subject=config.max_subjects,
                                                                      max_images_per_subject=config.max_emb_per_subject)

    # COX
    if config.eval_cox or config.save_cox_projections:
        paths, actual_issame = [], []
        proj_paths, proj_labels = [], []

        train_folds = [0, 1, 2]
        evaluation_folds = [3, 4, 5, 6, 7, 8, 9]

        for i, video_dir in enumerate(config.cox_video_name):

            cox_pairs_path = os.path.join(config.cox_pairs, video_dir, 'pairs.txt')
            cox_still_path = config.cox_still_dir
            cox_video_path = os.path.join(config.cox_video_dir, video_dir)

            cox_dataset = cox.cox_data(cox_still_path,
                                       cox_video_path,
                                       cox_pairs_path)

            if config.eval_cox:

                print('Fetching COX {} evaluation pairs.'.format(video_dir))

                paths, actual_issame = cox_dataset.get_pairs(evaluation_folds)

                cox_paths += paths
                cox_actual_issame += actual_issame

            if config.save_cox_projections:

                print('Fetching COX {} projections samples.'.format(video_dir))

                proj_paths, proj_labels = cox_dataset.get_paths_from_file(config.cox_projection,
                                                                          max_subject=config.max_subjects,
                                                                          max_images_per_subject=config.max_emb_per_subject,
                                                                          tag='_' + video_dir)

                cox_proj_paths += proj_paths
                cox_proj_labels += proj_labels

        del paths, actual_issame, proj_paths, proj_labels

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

            if projection_paths:
                save_embeddings(sess,
                                projection_paths,
                                label_batch,
                                embeddings,
                                image_paths_placeholder,
                                labels_placeholder,
                                batch_size_placeholder,
                                control_placeholder,
                                phase_train_placeholder,
                                eval_enqueue_op,
                                config.emb_dir)

            # Evaluate on LFW
            if lfw_paths:
                evaluate(sess, eval_enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder,
                         batch_size_placeholder, control_placeholder,
                         embeddings, label_batch, lfw_paths, lfw_actual_issame, config.batch_size, config.lfw_nrof_folds,
                         config.distance_metric, config.subtract_mean,
                         config.use_flipped_images, config.use_fixed_image_standardization, tag='LFW')

            # Evaluate on COX-S2V
            if cox_paths:
                evaluate(sess, eval_enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder,
                         batch_size_placeholder, control_placeholder,
                         embeddings, label_batch, cox_paths, cox_actual_issame, config.batch_size, config.lfw_nrof_folds,
                         config.distance_metric, config.subtract_mean,
                         config.use_flipped_images, config.use_fixed_image_standardization, tag='COX')
              
def evaluate(sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder, batch_size_placeholder, control_placeholder,
        embeddings, labels, image_paths, actual_issame, batch_size, nrof_folds, distance_metric, subtract_mean, use_flipped_images, use_fixed_image_standardization, tag='LFW'):
    # Run forward pass to calculate embeddings
    print('Runnning forward pass on ' + tag + ' images')
    
    # Enqueue one epoch of image paths and labels
    nrof_embeddings = len(actual_issame)*2  # nrof_pairs * nrof_images_per_pair
    nrof_flips = 2 if use_flipped_images else 1
    nrof_images = nrof_embeddings * nrof_flips
    labels_array = np.expand_dims(np.arange(0,nrof_images),1)
    image_paths_array = np.expand_dims(np.repeat(np.array(image_paths),nrof_flips),1)

    control_array = np.zeros_like(labels_array, np.int32)
    if use_fixed_image_standardization:
        control_array += np.ones_like(labels_array)*facenet.FIXED_STANDARDIZATION
    if use_flipped_images:
        # Flip every second image
        control_array += (labels_array % 2)*facenet.FLIP

    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array, control_placeholder: control_array})
    
    embedding_size = int(embeddings.get_shape()[1])
    assert nrof_images % batch_size == 0, 'The number of {} images ({}) must be an integer multiple of the batch size ({})'.format(tag, nrof_images, batch_size)
    nrof_batches = nrof_images // batch_size
    emb_array = np.zeros((nrof_images, embedding_size))
    lab_array = np.zeros((nrof_images,))
    for i in range(nrof_batches):
        feed_dict = {phase_train_placeholder:False, batch_size_placeholder:batch_size}
        emb, lab = sess.run([embeddings, labels], feed_dict=feed_dict)
        lab_array[lab] = lab
        emb_array[lab, :] = emb
        if i % 10 == 9:
            print('.', end='')
            sys.stdout.flush()
    print('')
    embeddings = np.zeros((nrof_embeddings, embedding_size*nrof_flips))
    if use_flipped_images:
        # Concatenate embeddings for flipped and non flipped version of the images
        embeddings[:,:embedding_size] = emb_array[0::2,:]
        embeddings[:,embedding_size:] = emb_array[1::2,:]
    else:
        embeddings = emb_array

    assert np.array_equal(lab_array, np.arange(nrof_images))==True, 'Wrong labels used for evaluation, possibly caused by training examples left in the input pipeline'
    tpr, fpr, accuracy, val, val_std, far, best_threshold, threshold_lowfar, tpr_lowfar, acc_lowfar = lfw.evaluate(embeddings, actual_issame, nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)

    print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    print('Best threshold: %1.3f' % best_threshold)
    print('Threshold: %1.3f @ FAR=%2.5f' % (threshold_lowfar, far))

    auc = metrics.auc(fpr, tpr)
    print('Area Under Curve (AUC): %1.6f' % auc)
    eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
    print('Equal Error Rate (EER): %1.6f' % eer)

def save_embeddings(sess, image_paths, labels, embeddings, image_paths_placeholder, labels_placeholder,
             batch_size_placeholder, control_placeholder, phase_train_placeholder, enqueue_op, log_dir):

    img_paths = image_paths
    nrof_images = len(img_paths)

    # Find batch size that fit the number of images between 1 - 100
    batch_size = 0
    if nrof_images < 100:
        batch_size = nrof_images
    else:
        for i in range(1,100):
            if nrof_images % i == 0:
                batch_size = i
    assert not batch_size == 0, 'Fitting batch size not found for {} images.'.format(nrof_images)
    print('Projection batch size: {}'.format(batch_size))

    # Run forward pass to calculate embeddings
    # Enqueue one epoch of image paths and labels

    labels_array = np.expand_dims(np.arange(0, nrof_images), 1)
    image_paths_array = np.expand_dims(np.array(img_paths), 1)
    control_array = np.zeros_like(labels_array, np.int32)

    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array,
                          control_placeholder: control_array})

    embedding_size = int(embeddings.get_shape()[1])
    nrof_batches = nrof_images // batch_size
    emb_array = np.zeros((nrof_images, embedding_size))
    lab_array = np.zeros((nrof_images,))
    for i in range(nrof_batches):
        feed_dict = {phase_train_placeholder: False, batch_size_placeholder: batch_size}
        emb, lab = sess.run([embeddings, labels], feed_dict=feed_dict)
        lab_array[lab] = lab
        emb_array[lab, :] = emb
        if i % 10 == 9:
            print('.', end='')
            sys.stdout.flush()
    print('')

    # Save embeddings for later projection
    print('Save {} embeddings for projection.'.format(len(emb_array)))

    with open(os.path.join(log_dir, 'emb.tsv'), "w") as emb_csv:
        csvWriter = csv.writer(emb_csv, delimiter='\t')
        csvWriter.writerows(emb_array)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str,
        help='Path to the configuration file', default='config/validation.ini')

    return parser.parse_args(argv)


if __name__ == '__main__':

    main(parse_arguments(sys.argv[1:]))
