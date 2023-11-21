# create data and label for CK+
#  0=anger 1=disgust, 2=fear, 3=happy, 4=sadness, 5=surprise, 6=contempt
# contain 135,177,75,207,84,249,54 images

import csv
import os
import numpy as np
import h5py
import skimage.io

ck_path = 'CK+48'

anger_path = os.path.join(ck_path, 'anger')
disgust_path = os.path.join(ck_path, 'disgust')
fear_path = os.path.join(ck_path, 'fear')
happy_path = os.path.join(ck_path, 'happy')
sadness_path = os.path.join(ck_path, 'sadness')
surprise_path = os.path.join(ck_path, 'surprise')
contempt_path = os.path.join(ck_path, 'contempt')

# # Creat the list to store the data and label information
data_x = []
data_y = []

datapath = os.path.join('data','CK_data.h5')
if not os.path.exists(os.path.dirname(datapath)):
    os.makedirs(os.path.dirname(datapath))

image_paths = [
    anger_path,
    disgust_path,
    fear_path,
    happy_path,
    sadness_path,
    surprise_path,
    contempt_path
]

# order the file, so the training set will not contain the test set (don't random)

for index, image_path in enumerate(image_paths):
    files = os.listdir(image_path)
    files.sort()
    for filename in files:
        I = skimage.io.imread(os.path.join(image_path,filename))
        data_x.append(I.tolist())
        data_y.append(index)

print(np.shape(data_x))
print(np.shape(data_y))

datafile = h5py.File(datapath, 'w')
datafile.create_dataset("data_pixel", dtype = 'uint8', data=data_x)
datafile.create_dataset("data_label", dtype = 'int64', data=data_y)
datafile.close()

print("Save data finish!!!")
