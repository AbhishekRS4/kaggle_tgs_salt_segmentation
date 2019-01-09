# @author : Abhishek R S

import os
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

img_net_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3)
img_net_std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3)

# read the json file and return the content
def read_config_file(json_file_name):
    # open and read the json file
    config = json.load(open(json_file_name))

    # return the content
    return config

# create the model directory if not present
def init(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# parse function for tensorflow dataset api
def parse_fn(img_name, lbl_name):
    img_string = tf.read_file(img_name)
    lbl_string = tf.read_file(lbl_name)

    img = tf.image.decode_png(img_string, channels=3)
    lbl = tf.image.decode_png(lbl_string, channels=0)

    img = tf.pad(img, paddings=[[0, 27], [0, 27], [0, 0]])
    lbl = tf.pad(lbl, paddings=[[0, 27], [0, 27], [0, 0]])

    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img - img_net_mean
    img = img / img_net_std

    lbl = tf.one_hot(lbl, depth=2)
    lbl = tf.squeeze(lbl)

    img = tf.transpose(img, perm=[2, 0, 1])
    lbl = tf.transpose(lbl, perm=[2, 0, 1])

    return img, lbl

# return tf dataset
def get_tf_dataset(images_list, labels_list, num_epochs, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((images_list, labels_list))
    dataset = dataset.shuffle(1000)
    dataset = dataset.map(parse_fn, num_parallel_calls=8)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.prefetch(1)

    return dataset

# split into train and test set
def get_train_valid_split(images_list, test_size=0.1, random_state=4):
    train_images_list, valid_images_list = train_test_split(
        images_list, test_size=test_size, random_state=random_state)
    train_images_list = shuffle(train_images_list)
    valid_images_list = shuffle(valid_images_list)

    return (train_images_list, valid_images_list)

# return the accuracy score of the predictions by the model
def get_accuracy_score(labels_groundtruth, labels_predicted):
    return accuracy_score(labels_groundtruth, labels_predicted)
