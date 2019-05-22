import os
from scipy import misc

def crop_center(img,cropx,cropy):
    y,x,ch = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]

image_dir = '/export/livia/data/lemoineh/ChokePoint/3DMM/3DMM-Fania'
output_dir = '/export/livia/data/lemoineh/ChokePoint/3DMM/3DMM-jpg'
image_size = 160

if not os.path.exists(output_dir):
    os.makedirs(output_dir)



subject_list = [dI for dI in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir,dI))]


for subject in subject_list:
    subject_path = os.path.join(image_dir, subject)
    out_subject_path = os.path.join(output_dir, subject)

    if not os.path.exists(out_subject_path):
        os.makedirs(out_subject_path)

    image_name_list = [f for f in os.listdir(subject_path) if os.path.isfile(os.path.join(subject_path, f))]

    for image_name in image_name_list:

        image_path = os.path.join(subject_path, image_name)

        img = misc.imread(image_path)
        width, height, ch = img.shape
        size = min(width, height)
        cropped = crop_center(img, size, size)
        scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        image_name_jpg = os.path.splitext(image_name)[0]+'.jpg'
        misc.imsave(os.path.join(out_subject_path, image_name_jpg), img, format="jpeg")

        print(image_name_jpg)