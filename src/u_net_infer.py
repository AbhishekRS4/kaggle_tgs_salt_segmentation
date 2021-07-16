# @author : Abhishek R S

import os
import cv2
import time
import numpy as np
import tensorflow as tf

from u_net_model as UNet
from u_net_utils import read_config_file, init

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
param_config_file_name = os.path.join(os.getcwd(), "u_net_config.json")
img_net_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3)
img_net_std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3)

# get softmax layer
def get_softmax_layer(logits, axis, name="softmax"):
    probs = tf.nn.softmax(logits, axis=axis, name=name)
    return probs

# parse function for tensorflow dataset api
def parse_fn(img_name):
    img_string = tf.read_file(img_name)
    img = tf.image.decode_png(img_string, channels=3)
    img = tf.pad(img, paddings=[[0, 27], [0, 27], [0, 0]])
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img - img_net_mean
    img = img / img_net_std
    img = tf.transpose(img, perm=[2, 0, 1])

    return img

# return tf dataset
def get_tf_dataset(images_list, num_epochs=1, batch_size=1):
    dataset = tf.data.Dataset.from_tensor_slices((images_list))
    dataset = dataset.map(parse_fn, num_parallel_calls=8)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.prefetch(1)

    return dataset

# run inference on test set
def infer():
    print("Reading the config file..................")
    config = read_config_file(param_config_file_name)
    model_directory = config["model_directory"] + str(config["num_epochs"])
    masks_directory = "masks_" + str(96)
    init(os.path.join(model_directory, masks_directory))
    print("Reading the Config File Completed........\n")

    print("Preparing test data.....................")
    test_list = os.listdir(config["test_images_path"])
    test_images_list = [os.path.join(config["test_images_path"], x) for x in test_list]
    num_test_samples = len(test_images_list)
    test_dataset = get_tf_dataset(test_images_list, 1, 1)
    iterator = test_dataset.make_one_shot_iterator()
    test_images = iterator.get_next()
    print("Preparing test data completed...........\n")

    print("Loading the Network.....................")
    axis = -1
    if config["data_format"] == "channels_first":
        axis = 1

    is_training = tf.placeholder(tf.bool)
    custom_u_net = UNet(config["num_kernels"], config["data_format"],
        is_training, config["densenet_path"], config["num_classes"])
    custom_u_net.densenet121_encoder(test_images)
    custom_u_net.custom_u_net()
    network_logits = custom_u_net.logits
    probs_prediction = get_softmax_layer(logits=network_logits, axis=axis)
    labels_prediction = tf.argmax(probs_prediction, axis=axis)
    print("Loading the Network Completed...........\n")

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    ss = tf.Session(config=tf.ConfigProto(device_count={"GPU": 1}))
    ss.run(tf.global_variables_initializer())

    # load the model parameters
    tf.train.Saver().restore(ss, os.path.join(os.getcwd(), model_directory, config["model_file"] + "-" + str(298)))

    print("Inference Started.......................")
    for img_file in test_images_list:
        ti = time.time()
        labels_predicted = ss.run(labels_prediction,
            feed_dict={is_training: not(config["training"])}
        )
        ti = time.time() - ti
        print(f"Time Taken for Inference : {ti}\n")

        labels_predicted_padded = np.transpose(labels_predicted, [1, 2, 0]).astype(np.uint8)
        labels_predicted_unpadded = labels_predicted_padded[:101, :101]
        cv2.imwrite(os.path.join(os.getcwd(), model_directory, masks_directory, "mask_" + img_file.split("/")[-1]), labels_predicted_unpadded)
    print("Inference Completed\n")
    ss.close()

def main():
    infer()

if __name__ == "__main__":
    main()
