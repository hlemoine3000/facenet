"""Training a face recognizer with TensorFlow based on the FaceNet paper
FaceNet: A Unified Embedding for Face Recognition and Clustering: http://arxiv.org/abs/1503.03832
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

from datetime import datetime
import os.path
import time
import sys
import tensorflow as tf
import numpy as np
import importlib
import itertools
import argparse
import csv
from src import facenet
from dataset import cox, lfw
from src import adversarialloss
from src import config_reader

from tensorflow.python.ops import data_flow_ops

from six.moves import xrange  # @UnresolvedImport
from shutil import copyfile

def main(args):

    config = config_reader.quadruplets_config(args.config)

    network = importlib.import_module(config.model_def)

    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(config.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(config.models_base_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)

    # Copy configuration to a text file
    copyfile(args.config, os.path.join(log_dir, 'config.txt'))

    np.random.seed(seed=config.seed)

    # Fetch datasets
    # VGGface2
    # print('Fetch VGGface2 as source dataset at {}'.format(config.vggface2_train_dir))
    # src_train_set = facenet.get_dataset(config.vggface2_train_dir)
    # validation_set = facenet.get_dataset(args.vggface2_val_dir)

    # COX S2V
    print('Fetch COX-S2V dataset.')

    cox_dataset = []
    for video in config.cox_video_name:

        cox_still_path = config.cox_still_dir
        cox_video_path = os.path.join(config.cox_video_dir, video)
        cox_pairs_path = os.path.join(config.cox_pairs, video, 'pairs.txt')

        cox_dataset.append(cox.cox_data(cox_still_path,
                                        cox_video_path,
                                        cox_pairs_path))

    train_folds = [0, 1, 2]
    evaluation_folds = [3, 4, 5, 6, 7, 8 ,9]

    src_vid2_train_set = cox_dataset[1].get_dataset(train_folds, video_only=True)
    src_vid4_train_set = cox_dataset[2].get_dataset(train_folds, video_only=True)

    src_train_set = cox.combine_dataset(src_vid2_train_set, src_vid4_train_set)
    tgt_train_set = cox_dataset[0].get_dataset(train_folds, video_only=True)

    cox_vid1_paths, cox_vid1_issame = cox_dataset[0].get_pairs(evaluation_folds)
    cox_vid2_paths, cox_vid2_issame = cox_dataset[1].get_pairs(evaluation_folds)
    cox_vid4_paths, cox_vid4_issame = cox_dataset[2].get_pairs(evaluation_folds)

    # cox_paths = cox_paths[:-8]
    # cox_issame = cox_issame[:-4]

    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)
    if config.pretrained_model:
        print('Pre-trained model: %s' % os.path.expanduser(config.pretrained_model))

    if config.lfw_dir:
        print('LFW directory: %s' % config.lfw_dir)
        # Read the file containing the pairs used for testing
        pairs = lfw.read_pairs(os.path.expanduser(config.lfw_pairs))
        # Get the paths for the corresponding images
        lfw_paths, lfw_actual_issame = lfw.get_paths(os.path.expanduser(config.lfw_dir), pairs)
        # Get the paths for embeddings projection

    # Get the paths for embeddings projection

    lfw_proj_paths, lfw_proj_labels = lfw.get_paths_from_file(config.lfw_dir, config.lfw_projection)

    cox_proj_paths, cox_proj_labels = [], []

    for dataset in cox_dataset:
        proj_paths, proj_labels = dataset.get_paths_from_file(config.cox_projection)
        cox_proj_paths += proj_paths
        cox_proj_labels += proj_labels


    # Combine projection paths
    projection_paths = lfw_proj_paths + cox_proj_paths
    proj_labels = lfw_proj_labels + cox_proj_labels

    # Create label map if does not exist
    metadata_filename = 'meta.tsv'
    emb_dir = os.path.join(os.path.expanduser(log_dir), 'emb')
    if not os.path.isdir(emb_dir):  # Create the log directory if it doesn't exist
        os.makedirs(emb_dir)
    with open(os.path.join(emb_dir, metadata_filename), "w") as meta_file:
        csvWriter = csv.writer(meta_file, delimiter='\t')
        csvWriter.writerows(np.array([proj_labels]).T)
    print('Embeddings meta data saved at {}'.format(os.path.join(emb_dir, metadata_filename)))

    with tf.Graph().as_default():
        tf.set_random_seed(config.seed)
        global_step = tf.Variable(0, trainable=False)

        # Placeholder for the learning rate
        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')

        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')

        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

        image_paths_placeholder = tf.placeholder(tf.string, shape=(None, 4), name='image_paths')
        labels_placeholder = tf.placeholder(tf.int64, shape=(None, 4), name='labels')

        input_queue = data_flow_ops.FIFOQueue(capacity=100000,
                                              dtypes=[tf.string, tf.int64],
                                              shapes=[(4,), (4,)],
                                              shared_name=None, name=None)
        enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder])

        nrof_preprocess_threads = 4
        images_and_labels = []
        for _ in range(nrof_preprocess_threads):
            filenames, label = input_queue.dequeue()
            images = []
            for filename in tf.unstack(filenames):
                file_contents = tf.read_file(filename)
                image = tf.image.decode_image(file_contents, channels=3)

                if config.random_crop:
                    image = tf.random_crop(image, [config.image_size, config.image_size, 3])
                else:
                    image = tf.image.resize_image_with_crop_or_pad(image, config.image_size, config.image_size)
                if config.random_flip:
                    image = tf.image.random_flip_left_right(image)

                # pylint: disable=no-member
                image.set_shape((config.image_size, config.image_size, 3))
                images.append(tf.image.per_image_standardization(image))
            images_and_labels.append([images, label])

        image_batch, labels_batch = tf.train.batch_join(
            images_and_labels, batch_size=batch_size_placeholder,
            shapes=[(config.image_size, config.image_size, 3), ()], enqueue_many=True,
            capacity=4 * nrof_preprocess_threads * config.batch_size,
            allow_smaller_final_batch=True)
        image_batch = tf.identity(image_batch, 'image_batch')
        image_batch = tf.identity(image_batch, 'input')
        labels_batch = tf.identity(labels_batch, 'label_batch')

        # Build the inference graph
        prelogits, _ = network.inference(image_batch, config.keep_probability,
                                         phase_train=phase_train_placeholder, bottleneck_layer_size=config.embedding_size,
                                         weight_decay=config.weight_decay)

        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
        # Split embeddings into anchor, positive and negative and calculate triplet loss
        anchor, positive, negative, target = tf.unstack(tf.reshape(embeddings, [-1, 4, config.embedding_size]), 4, 1)
        loss, adv_loss, triplet_loss = adversarialloss.quadruplets_loss(anchor, positive, negative, target, config.alpha,
                                                                        config.lamb, config.zeta)
        # triplet_loss = tripletloss.triplet_loss(anchor, positive, negative, args.alpha)
        # adv_loss = adversarialloss.adversarial_loss(anchor, target, args.alpha)

        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
                                                   config.learning_rate_decay_epochs * config.epoch_size,
                                                   config.learning_rate_decay_factor, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        # Calculate the total losses
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([loss] + regularization_losses, name='total_loss')  # + [adv_loss]
        reg_loss = tf.add_n(regularization_losses, name='total_loss')

        # Build a Graph that trains the model with one batch of examples and updates the model parameters
        train_op = facenet.train(total_loss, global_step, config.optimizer,
                                 learning_rate, config.moving_average_decay, tf.global_variables())



        # Create a saver
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # Initialize variables
        sess.run(tf.global_variables_initializer(), feed_dict={phase_train_placeholder: True})
        sess.run(tf.local_variables_initializer(), feed_dict={phase_train_placeholder: True})

        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)

        model = facenet.model_container(enqueue_op=enqueue_op,
                                        embeddings=embeddings,
                                        labels_batch=labels_batch,
                                        train_op=train_op,
                                        summary_op=summary_op,
                                        summary_writer=summary_writer,
                                        total_loss=total_loss,
                                        triplet_loss=triplet_loss,
                                        adv_loss=adv_loss,
                                        image_paths_placeholder=image_paths_placeholder,
                                        labels_placeholder=labels_placeholder,
                                        batch_size_placeholder=batch_size_placeholder,
                                        learning_rate_placeholder=learning_rate_placeholder,
                                        phase_train_placeholder=phase_train_placeholder,
                                        learning_rate=config.learning_rate,
                                        batch_size=config.batch_size,
                                        embedding_size=config.embedding_size)

        with sess.as_default():

            if config.pretrained_model:
                print('Restoring pretrained model: %s' % config.pretrained_model)
                saver.restore(sess, os.path.expanduser(config.pretrained_model))

            # Training and validation loop
            epoch = 0
            while epoch < config.max_nrof_epochs:
                step = sess.run(global_step, feed_dict=None)
                epoch = step // config.epoch_size

                save_embeddings(config,
                                model,
                                sess,
                                projection_paths,
                                log_dir,
                                epoch)

                # Evaluate on COX

                evaluate(config,
                         model,
                         sess,
                         cox_vid1_paths,
                         cox_vid1_issame,
                         log_dir,
                         step,
                         tag='cox_video1')

                evaluate(config,
                         model,
                         sess,
                         cox_vid2_paths,
                         cox_vid2_issame,
                         log_dir,
                         step,
                         tag='cox_video2')

                evaluate(config,
                         model,
                         sess,
                         cox_vid4_paths,
                         cox_vid4_issame,
                         log_dir,
                         step,
                         tag='cox_video4')

                if config.lfw_dir:
                    evaluate(config,
                             model,
                             sess,
                             lfw_paths,
                             lfw_actual_issame,
                             log_dir,
                             step,
                             tag='lfw')

                # Train for one epoch
                train(config,
                      model,
                      sess,
                      src_train_set,
                      tgt_train_set,
                      epoch,
                      global_step)


                # Save variables and the metagraph if it doesn't exist already
                save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, step)

                # Evaluate on LFW

    return model_dir

def train(config: config_reader.quadruplets_config,
          model: facenet.model_container,
          sess,
          src_dataset,
          tgt_dataset,
          epoch,
          global_step):
    batch_number = 0
    # if config.learning_rate > 0.0:
    #     lr = config.learning_rate
    # else:
    #     lr = facenet.get_learning_rate_from_file(learning_rate_schedule_file, epoch)


    while batch_number < config.epoch_size:
        # Sample people randomly from the dataset
        src_still_paths, src_video_paths = [], []
        if isinstance(src_dataset[0], cox.COX_ImageClass):
            src_video_paths, src_still_paths, src_num_per_class = cox.sample_people(src_dataset,
                                                                                    config.people_per_batch,
                                                                                    config.images_per_person)
        elif isinstance(src_dataset[0], cox.ImageClass):
            src_video_paths, src_num_per_class = sample_people(src_dataset,
                                                               config.people_per_batch,
                                                               config.images_per_person)
        else:
            raise ValueError('Source dataset class do not fit.')

        tgt_image_paths, tgt_num_per_class = sample_people(tgt_dataset,
                                                           config.people_per_batch,
                                                           config.images_per_person)

        # Source still samples forward pass
        if src_still_paths:
            print('Running forward pass on source still sample images: ', end='')
            start_time = time.time()
            src_still_emb = forward_pass(sess, model, src_still_paths)
            print('%.3f' % (time.time() - start_time))

        # Source video samples forward pass
        print('Running forward pass on source video sample images: ', end='')
        start_time = time.time()
        src_video_emb = forward_pass(sess, model, src_video_paths)
        print('%.3f' % (time.time() - start_time))

        # Target video samples forward pass
        print('Running forward pass on source still sample images: ', end='')
        start_time = time.time()
        tgt_video_emb = forward_pass(sess, model, tgt_image_paths)
        print('%.3f' % (time.time() - start_time))

        # Select triplets based on the embeddings
        print('Selecting suitable quadruplets for training')
        if src_still_paths:
            quadruplets, nrof_random_negs, nrof_quadruplets = adversarialloss.select_cox_quadruplets(src_still_emb,
                                                                                                     src_video_emb,
                                                                                                     tgt_video_emb,
                                                                                                     src_still_paths,
                                                                                                     src_video_paths,
                                                                                                     tgt_image_paths,
                                                                                                     src_num_per_class,
                                                                                                     config.people_per_batch,
                                                                                                     config.alpha)
        else:
            quadruplets, nrof_random_negs, nrof_quadruplets = adversarialloss.select_quadruplets(src_video_emb,
                                                                                                 tgt_video_emb,
                                                                                                 src_video_paths,
                                                                                                 tgt_image_paths,
                                                                                                 src_num_per_class,
                                                                                                 config.people_per_batch,
                                                                                                 config.alpha)

        selection_time = time.time() - start_time
        print('(nrof_random_negs, nrof_quadruplets) = (%d, %d): time=%.3f seconds' %
              (nrof_random_negs, nrof_quadruplets, selection_time))

        # Perform training on the selected quadruplets
        nrof_batches = int(np.ceil(nrof_quadruplets * 4 / config.batch_size))
        quadruplets_paths = list(itertools.chain(*quadruplets))
        labels_array = np.reshape(np.arange(len(quadruplets_paths)), (-1, 4))
        quadruplets_paths_array = np.reshape(np.expand_dims(np.array(quadruplets_paths), 1), (-1, 4))
        sess.run(model.enqueue_op, {model.image_paths_placeholder: quadruplets_paths_array, model.labels_placeholder: labels_array})
        nrof_examples = len(quadruplets_paths)
        train_time = 0
        i = 0
        emb_array = np.zeros((nrof_examples, model.embedding_size))
        loss_array = np.zeros((nrof_quadruplets,))
        summary = tf.Summary()
        step = 0
        while i < nrof_batches:
            start_time = time.time()
            batch_size = min(nrof_examples - i * config.batch_size, config.batch_size)
            feed_dict = {model.batch_size_placeholder: batch_size,
                         model.learning_rate_placeholder: model.learning_rate,
                         model.phase_train_placeholder: True}
            triplet_err, adv_err, total_err, _, step, emb, lab, summary_res = sess.run(
                [model.triplet_loss,
                 model.adv_loss,
                 model.total_loss,
                 model.train_op,
                 global_step,
                 model.embeddings,
                 model.labels_batch,
                 model.summary_op],
                feed_dict=feed_dict)
            # emb_array[lab,:] = emb
            # loss_array[i] = err
            duration = time.time() - start_time
            print('Epoch: [%d][%d/%d]\tTime %.3f\nTotal loss %2.3f\nTriplet loss %2.3f\nAdv loss %2.3f' %
                  (epoch, batch_number + 1, config.epoch_size, duration, total_err, triplet_err, adv_err))
            batch_number += 1
            i += 1
            train_time += duration
            summary.value.add(tag='loss/total_loss', simple_value=total_err)
            summary.value.add(tag='loss/triplet_loss', simple_value=triplet_err)
            summary.value.add(tag='loss/adv_loss', simple_value=adv_err)
            # summary.value.add(tag='loss/regularisationL2_err', simple_value=reg_err)

        # Add validation loss and accuracy to summary
        # pylint: disable=maybe-no-member
        summary.value.add(tag='time/selection', simple_value=selection_time)
        model.summary_writer.add_summary(summary, step)

    return step

def forward_pass(sess,
                 model: facenet.model_container,
                 image_paths):
    nrof_examples = len(image_paths)
    labels_array = np.reshape(np.arange(nrof_examples), (-1, 4))
    image_paths_array = np.reshape(np.expand_dims(np.array(image_paths), 1), (-1, 4))
    sess.run(model.enqueue_op, {model.image_paths_placeholder: image_paths_array, model.labels_placeholder: labels_array})
    emb_array = np.zeros((nrof_examples, model.embedding_size))
    nrof_batches = int(np.ceil(nrof_examples / model.batch_size))
    for i in range(nrof_batches):
        batch_size = min(nrof_examples - i * model.batch_size, model.batch_size)
        emb, lab = sess.run([model.embeddings, model.labels_batch], feed_dict={model.batch_size_placeholder: batch_size,
                                                                                     model.learning_rate_placeholder: model.learning_rate,
                                                                                     model.phase_train_placeholder: True})
        emb_array[lab, :] = emb

    return emb_array


def sample_people(dataset,
                  people_per_batch,
                  images_per_person):

    nrof_images = people_per_batch * images_per_person

    # Sample classes from the dataset
    nrof_classes = len(dataset)
    class_indices = np.arange(nrof_classes)
    np.random.shuffle(class_indices)

    i = 0
    image_paths = []
    num_per_class = []
    sampled_class_indices = []
    # Sample images from these classes until we have enough
    while len(image_paths) < nrof_images:
        class_index = class_indices[i]
        nrof_images_in_class = len(dataset[class_index])
        image_indices = np.arange(nrof_images_in_class)
        np.random.shuffle(image_indices)
        nrof_images_from_class = min(nrof_images_in_class, images_per_person, nrof_images - len(image_paths))
        idx = image_indices[0:nrof_images_from_class]
        image_paths_for_class = [dataset[class_index].image_paths[j] for j in idx]
        sampled_class_indices += [class_index] * nrof_images_from_class
        image_paths += image_paths_for_class
        num_per_class.append(nrof_images_from_class)
        i += 1

    return image_paths, num_per_class


def evaluate(config: config_reader.quadruplets_config,
             model: facenet.model_container,
             sess,
             image_paths,
             actual_issame,
             log_dir,
             step,
             tag='eval'):

    result = {}
    start_time = time.time()
    # Run forward pass to calculate embeddings
    print('Running forward pass on ' + tag + ' images: ', end='')

    nrof_images = len(actual_issame) * 2
    assert (len(image_paths) == nrof_images)
    labels_array = np.reshape(np.arange(nrof_images), (-1, 4))
    image_paths_array = np.reshape(np.expand_dims(np.array(image_paths), 1), (-1, 4))
    sess.run(model.enqueue_op, {model.image_paths_placeholder: image_paths_array, model.labels_placeholder: labels_array})
    emb_array = np.zeros((nrof_images, model.embedding_size))
    nrof_batches = int(np.ceil(nrof_images / model.batch_size))
    label_check_array = np.zeros((nrof_images,))
    for i in xrange(nrof_batches):
        batch_size = min(nrof_images - i * model.batch_size, model.batch_size)
        emb, lab = sess.run([model.embeddings, model.labels_batch], feed_dict={model.batch_size_placeholder: model.batch_size,
                                                                               model.learning_rate_placeholder: 0.0,
                                                                               model.phase_train_placeholder: False})
        emb_array[lab, :] = emb
        label_check_array[lab] = 1
    print('%.3f' % (time.time() - start_time))

    assert (np.all(label_check_array == 1))

    tpr, fpr, accuracy, val, val_std, far, best_threshold, threshold_lowfar, tpr_lowfar, acc_lowfar = lfw.evaluate(
        emb_array, actual_issame, nrof_folds=config.nrof_folds)

    print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    lfw_time = time.time() - start_time
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    # pylint: disable=maybe-no-member
    summary.value.add(tag=tag + '/accuracy', simple_value=np.mean(accuracy))
    summary.value.add(tag=tag + '/val_rate', simple_value=val)
    summary.value.add(tag=tag + '/best_threshold', simple_value=best_threshold)
    summary.value.add(tag=tag + '/val_rate_threshold', simple_value=threshold_lowfar)
    summary.value.add(tag='time/' + tag, simple_value=lfw_time)
    model.summary_writer.add_summary(summary, step)
    with open(os.path.join(log_dir, tag + '_result.txt'), 'at') as f:
        f.write('%d\t%.5f\t%.5f\n' % (step, np.mean(accuracy), val))

    result['accuracy'] = np.mean(accuracy)
    result['val_rate'] = val
    result['best_threshold'] = best_threshold

    return result

def save_embeddings(config: config_reader.quadruplets_config,
                    model: model_container,
                    sess,
                    image_paths,
                    log_dir,
                    epoch,
                    tag='projection_set'):

    img_paths = image_paths

    # Fill the image extractor with dumb images if necessary
    # There are 4 parallel image extractor
    nrof_images = len(img_paths)
    num_missing = 4 - (nrof_images % 4)
    if not num_missing == 0:

        for i in range(0, num_missing):
            img_paths.append(img_paths[0])
        nrof_images = len(img_paths)


    start_time = time.time()
    # Run forward pass to calculate embeddings
    print('Running forward pass ' + tag + ' images: ', end='')

    labels_array = np.reshape(np.arange(nrof_images), (-1, 4))
    image_paths_array = np.reshape(np.expand_dims(np.array(img_paths), 1), (-1, 4))
    sess.run(model.enqueue_op, {model.image_paths_placeholder: image_paths_array, model.labels_placeholder: labels_array})
    emb_array = np.zeros((nrof_images, model.embedding_size))
    nrof_batches = int(np.ceil(nrof_images / model.batch_size))
    label_check_array = np.zeros((nrof_images,))
    for i in xrange(nrof_batches):
        batch_size = min(nrof_images - i * model.batch_size, model.batch_size)
        emb, lab = sess.run([model.embeddings, model.labels_batch], feed_dict={model.batch_size_placeholder: model.batch_size,
                                                                               model.learning_rate_placeholder: 0.0,
                                                                               model.phase_train_placeholder: False})
        emb_array[lab, :] = emb
        label_check_array[lab] = 1

    if num_missing != 0:
        emb_array = emb_array[:-num_missing]

    print('%.3f' % (time.time() - start_time))

    assert (np.all(label_check_array == 1))

    # Save embeddings for later projection
    print('Save {} embeddings for projection.'.format(len(emb_array)))

    with open(os.path.join(log_dir, 'emb{}.tsv'.format(epoch)), "a") as emb_csv:
        csvWriter = csv.writer(emb_csv, delimiter='\t')
        csvWriter.writerows(emb_array)


def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    # pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)


def get_learning_rate_from_file(filename, epoch):
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split('#', 1)[0]
            if line:
                par = line.strip().split(':')
                e = int(par[0])
                lr = float(par[1])
                if e <= epoch:
                    learning_rate = lr
                else:
                    return learning_rate

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str,
        help='Path to the configuration file', default='config/train_quadruplets_cox.ini')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
