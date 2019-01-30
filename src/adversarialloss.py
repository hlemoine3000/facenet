import numpy as np
import tensorflow as tf

from six.moves import xrange  # @UnresolvedImport

def select_quadruplets(src_embeddings, tgt_embeddings, src_image_paths, tgt_image_paths, nrof_images_per_class, people_per_batch, alpha):
    """ Select the triplets for training
    """
    trip_idx = 0
    emb_start_idx = 0
    num_quads = 0
    quadruplets = []

    # VGG Face: Choosing good triplets is crucial and should strike a balance between
    #  selecting informative (i.e. challenging) examples and swamping training with examples that
    #  are too hard. This is achieve by extending each pair (a, p) to a triplet (a, p, n) by sampling
    #  the image n at random, but only between the ones that violate the triplet loss margin. The
    #  latter is a form of hard-negative mining, but it is not as aggressive (and much cheaper) than
    #  choosing the maximally violating example, as often done in structured output learning.

    # For each person in this batch
    for i in xrange(people_per_batch): # 720
        nrof_images = int(nrof_images_per_class[i])
        for j in xrange(1, nrof_images):
            a_idx = emb_start_idx + j - 1
            neg_dists_sqr = np.sum(np.square(src_embeddings[a_idx] - src_embeddings), 1)
            for pair in xrange(j, nrof_images):  # For every possible positive pair.
                p_idx = emb_start_idx + pair
                pos_dist_sqr = np.sum(np.square(src_embeddings[a_idx] - src_embeddings[p_idx]))
                neg_dists_sqr[emb_start_idx:emb_start_idx + nrof_images] = np.NaN

                # all_neg = np.where(np.logical_and(neg_dists_sqr-pos_dist_sqr<alpha, pos_dist_sqr<neg_dists_sqr))[0]  # FaceNet selection
                all_neg = np.where(neg_dists_sqr - pos_dist_sqr < alpha)[0]  # VGG Face selecction

                nrof_random_negs = all_neg.shape[0]
                if nrof_random_negs > 0:
                    rnd_idx = np.random.randint(nrof_random_negs)
                    n_idx = all_neg[rnd_idx]
                    tgt_idx = np.random.randint(len(tgt_image_paths))
                    quadruplets.append((src_image_paths[a_idx], src_image_paths[p_idx], src_image_paths[n_idx], tgt_image_paths[tgt_idx]))
                    # print('Triplet %d: (%d, %d, %d), pos_dist=%2.6f, neg_dist=%2.6f (%d, %d, %d, %d, %d)' %
                    #    (trip_idx, a_idx, p_idx, n_idx, pos_dist_sqr, neg_dists_sqr[n_idx], nrof_random_negs, rnd_idx, i, j, emb_start_idx))
                    trip_idx += 1

                num_quads += 1

        emb_start_idx += nrof_images

    np.random.shuffle(quadruplets)
    return quadruplets, num_quads, len(quadruplets)

def select_cox_quadruplets(src_still_embeddings,
                           src_video_embeddings,
                           tgt_embeddings,
                           src_still_image_paths,
                           src_video_image_paths,
                           tgt_image_paths,
                           nrof_images_per_class,
                           people_per_batch,
                           alpha):
    """ Select the triplets for training
    """
    trip_idx = 0
    emb_start_idx = 0
    num_quads = 0
    quadruplets = []

    # VGG Face: Choosing good triplets is crucial and should strike a balance between
    #  selecting informative (i.e. challenging) examples and swamping training with examples that
    #  are too hard. This is achieve by extending each pair (a, p) to a triplet (a, p, n) by sampling
    #  the image n at random, but only between the ones that violate the triplet loss margin. The
    #  latter is a form of hard-negative mining, but it is not as aggressive (and much cheaper) than
    #  choosing the maximally violating example, as often done in structured output learning.

    for i in xrange(people_per_batch):
        nrof_images = int(nrof_images_per_class[i])
        for j in xrange(1, nrof_images):

            a_idx = i
            neg_dists_sqr = np.sum(np.square(src_still_embeddings[a_idx] - src_video_embeddings), 1)
            for pair in xrange(j, nrof_images):  # For every possible positive pair.
                p_idx = emb_start_idx + pair
                pos_dist_sqr = np.sum(np.square(src_still_embeddings[a_idx] - src_video_embeddings[p_idx]))
                neg_dists_sqr[emb_start_idx:emb_start_idx + nrof_images] = np.NaN
                all_neg = np.where(np.logical_and(neg_dists_sqr-pos_dist_sqr<alpha, pos_dist_sqr<neg_dists_sqr))[0]  # FaceNet selection
                #all_neg = np.where(neg_dists_sqr - pos_dist_sqr < alpha)[0]  # VGG Face selecction
                nrof_random_negs = all_neg.shape[0]
                if nrof_random_negs > 0:
                    rnd_idx = np.random.randint(nrof_random_negs)
                    n_idx = all_neg[rnd_idx]
                    tgt_idx = np.random.randint(len(tgt_image_paths))
                    quadruplets.append((src_still_image_paths[a_idx],
                                        src_video_image_paths[p_idx],
                                        src_video_image_paths[n_idx],
                                        tgt_image_paths[tgt_idx]))
                    # print('Triplet %d: (%d, %d, %d), pos_dist=%2.6f, neg_dist=%2.6f (%d, %d, %d, %d, %d)' %
                    #    (trip_idx, a_idx, p_idx, n_idx, pos_dist_sqr, neg_dists_sqr[n_idx], nrof_random_negs, rnd_idx, i, j, emb_start_idx))
                    trip_idx += 1

                num_quads += 1

        emb_start_idx += nrof_images

    np.random.shuffle(quadruplets)
    return quadruplets, num_quads, len(quadruplets)

def quadruplets_loss(anchor, positive, negative, target, alpha, lamb, zeta=1.0):
    """Calculate the triplet loss according to the FaceNet paper

    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.

    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    with tf.variable_scope('quadruplet_loss'):
        # Original
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
        tgt_dist = tf.reduce_sum(tf.square(tf.subtract(target, negative)), 1)

        triplet_loss = tf.multiply(tf.add(tf.subtract(pos_dist, neg_dist), alpha), zeta)

        adv_loss = tf.multiply(tf.add(tf.subtract(pos_dist, tgt_dist), alpha/2), lamb)

        total_loss = tf.add(triplet_loss, adv_loss)

        loss = tf.reduce_mean(tf.maximum(total_loss, 0.0), 0)

        scalar_triplet_loss = tf.reduce_mean(tf.maximum(triplet_loss, 0.0), 0)
        scalar_adv_loss = tf.reduce_mean(tf.maximum(adv_loss, 0.0), 0)

    return loss, scalar_adv_loss, scalar_triplet_loss

def quadruplets_loss(anchor, positive, negative, target, alpha, lamb, zeta=1.0):
    """Calculate the triplet loss according to the FaceNet paper

    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.

    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    with tf.variable_scope('quadruplet_loss'):
        # Triplet loss
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
        # Adversarial loss
        tgt_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, target)), 1)

        triplet_loss = tf.multiply(tf.add(tf.subtract(pos_dist, neg_dist), alpha), zeta)

        adv_loss = tf.multiply(tgt_dist, lamb)

        total_loss = tf.add(triplet_loss, adv_loss)

        loss = tf.reduce_mean(tf.maximum(total_loss, 0.0), 0)

        scalar_triplet_loss = tf.reduce_mean(tf.maximum(triplet_loss, 0.0), 0)
        scalar_adv_loss = tf.reduce_mean(tf.maximum(adv_loss, 0.0), 0)

    return loss, scalar_adv_loss, scalar_triplet_loss

def adversarial_loss(source, target, alpha):
    """Calculate the target distance loss

    Args:
      anchor: the embeddings for the anchor images.
      target: the embeddings for the positive images.

    Returns:
      the adversarial loss.
    """
    with tf.variable_scope('adversarial_loss'):
        # Original
        dist = tf.reduce_sum(tf.square(tf.subtract(source, target)), 1)
        loss = tf.reduce_mean(tf.maximum(dist, 0.0), 0)

    return loss