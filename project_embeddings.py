

import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

def main(args):

    emb_path = os.path.join(args.logs_dir, 'embeddings.csv')
    labels_path = os.path.join(args.logs_dir, 'emb_labels.csv')

    assert os.path.isfile(emb_path), 'Embeddings not found at: {}'.format(emb_path)
    assert os.path.isfile(labels_path), 'Labels not found at: {}'.format(labels_path)

    labels_data = []
    num_images = 0
    with open(labels_path) as labels_csv_file:
        for line in labels_csv_file:
            subject, num = line.strip().split(',')
            labels_data.append([subject, int(num)])
            num_images += int(num)

    epoch = 0

    with open(emb_path, "rb") as emb_csv_file:

        emb_array = np.loadtxt(emb_csv_file, delimiter=",", skiprows=1)

        # row_count = sum(1 for row in csv_reader)
        # num_epoch = row_count // num_images

    start_idx = epoch * num_images
    end_idx = start_idx + num_images
    select_emb_array = emb_array[start_idx: end_idx]


    arr = np.empty((0, 300), dtype='f')
    word_labels = [word]

    # get close words
    close_words = model.similar_by_word(word)

    # add the vector for each of the closest words to the array
    arr = np.append(arr, np.array([model[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)

    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    # display scatter plot
    plt.scatter(x_coords, y_coords)

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min() + 0.00005, x_coords.max() + 0.00005)
    plt.ylim(y_coords.min() + 0.00005, y_coords.max() + 0.00005)
    plt.show()

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('logs_dir', type=str, help='Directory with the embeddings logged')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))