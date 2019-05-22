
import os
import numpy as np

# import facenet
class ImageClass():
    "Stores the paths to images for a given class"

    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)

    def __add__(self, other):
        self.image_paths += other.image_paths
        return self


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

    def __add__(self, other):
        self.video_paths += other.video_paths
        return self

class S2V_ImageClass():
    "Stores the paths to images for a given class"

    def __init__(self, name, video_paths, still_paths):
        self.name = name
        self.video_paths = video_paths
        self.still_paths = still_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.video_paths)) + ' images'

    def __len__(self):
        return len(self.video_paths)

    def __add__(self, other):
        self.video_paths += other.video_paths
        self.still_paths += other.still_paths
        return self

def get_image_paths(image_dir):
    image_paths = []
    if os.path.isdir(image_dir):
        images = os.listdir(image_dir)
        image_paths = [os.path.join(image_dir,img) for img in images]
    return image_paths

def add_extension(path):
    if os.path.exists(path+'.jpg'):
        return path+'.jpg'
    if os.path.exists(path+'.JPG'):
        return path+'.JPG'
    elif os.path.exists(path+'.png'):
        return path+'.png'
    else:
        raise RuntimeError('No file "%s" with extension png or jpg.' % path)

def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)