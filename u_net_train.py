# @author : Abhishek R S

import math
import os
import time
import numpy as np
import tensorflow as tf
from u_net_utils import init, read_config_file, get_train_valid_split, get_tf_dataset
import network_architecture as na
from lovasz_loss import lovasz_softmax

param_config_file_name = os.path.join(os.getcwd(), 'u_net_config.json')

# return the softmax layer
def get_softmax_layer(input_tensor, axis=-1, name='softmax'):
    prediction = tf.nn.softmax(input_tensor, dim=axis, name=name)
    return prediction

# return the sorensen-dice coefficient
def dice_loss(ground_truth, predicted_logits, axis=-1, smooth=1e-5, name='mean_dice_loss'):
    predicted_probs = get_softmax_layer(
        input_tensor=predicted_logits, axis=axis)
    predicted_class = tf.round(predicted_probs)
    intersection = tf.reduce_sum(tf.multiply(
        ground_truth, predicted_class), axis=[1, 2, 3])
    union = tf.reduce_sum(ground_truth, axis=[
                          1, 2, 3]) + tf.reduce_sum(predicted_class, axis=[1, 2, 3])
    dice_coeff = (2. * intersection + smooth) / (union + smooth)
    # use sorensen dice coeff

    #dice_loss = tf.reduce_mean(-tf.log(dice_coeff), name = name)
    dice_loss = tf.reduce_mean(-tf.log(dice_coeff), name=name)
    return dice_loss

# return cross entropy loss
def cross_entropy_loss(ground_truth, prediction, axis, name='mean_cross_entropy'):
    mean_ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=ground_truth, logits=prediction, dim=axis), name=name)
    return mean_ce

# return the optimizer which has to be used to minimize the loss function
def get_optimizer_2(global_step_pl, loss_function):
    initial_learning_rate = 0.001
    decay_steps = 300
    end_learning_rate = 0.000001
    decay_rate = 0.97
    power = 0.95
    learning_rate = tf.train.polynomial_decay(
        initial_learning_rate, global_step_pl, decay_steps, end_learning_rate, power=power)

    adam_optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate, epsilon=0.0001).minimize(loss_function)
    return adam_optimizer

# return the optimizer which has to be used to minimize the loss function
def get_optimizer(learning_rate, loss_function):
    adam_optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(loss_function)
    return adam_optimizer

# save the trained model
def save_model(session, model_directory, model_file, epoch):
    saver = tf.train.Saver()
    saver.save(session, os.path.join(os.getcwd(), os.path.join(
        model_directory, model_file)), global_step=(epoch + 1))

# start batch training of the network
def batch_train():

    print('Reading the config file..................')
    config = read_config_file(param_config_file_name)
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    print('Reading the config file completed........')
    print('')

    print('Initializing.............................')
    model_directory = config['model_directory'] + str(config['num_epochs'])
    init(model_directory)
    print('Initializing completed...................')
    print('')

    print('Preparing train data.....................')
    all_data_list = os.listdir(config['train_images_path'])

    images_list = [os.path.join(config['train_images_path'], x)
                   for x in all_data_list]
    labels_list = [os.path.join(config['train_labels_path'], x)
                   for x in all_data_list]

    train_ratio = 0.8

    num_train_samples = int(len(images_list) * train_ratio)
    # 0.8 * 8000 = 6400
    num_train_batches = int(math.ceil(num_train_samples / float(batch_size)))
    # 200

    num_valid_samples = len(images_list) - num_train_samples
    # 8000 - 6400 = 1600
    num_valid_batches = int(math.ceil(num_valid_samples / float(batch_size)))
    # 50

    print('Preparing train data completed...........')
    print('')

    print('Building the network.....................')
    axis = -1
    order = 'BHWC'
    if config['data_format'] == 'channels_first':
        axis = 1
        order = 'BCHW'

    training_pl = tf.placeholder(tf.bool)
    dataset = get_tf_dataset(images_list, labels_list, num_epochs, batch_size)

    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()

    net_arch = na.UNet(config['num_kernels'], config['data_format'],
                       training_pl, config['densenet_path'], config['num_classes'])

    #net_arch.custom_u_net(tf.placeholder(shape = [None, 3, 101, 101], dtype = tf.float32, name = 'input'))
    net_arch.densenet121_encoder(features)
    net_arch.custom_u_net()
    logits = net_arch.logits

    train_var_list = [v for v in tf.trainable_variables()]
    weight_decay = 0.0001

    '''
    loss_1 = dice_loss(labels, logits, axis = axis)
    loss_2 = cross_entropy_loss(labels, logits, axis = axis)
    loss_3 = weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in train_var_list]) 
    loss = loss_1 + loss_2 + loss_3
    optimizer_op = get_optimizer(config['learning_rate'], loss)
    '''

    loss_1 = lovasz_softmax(logits, tf.argmax(labels, axis=axis), order=order)
    loss_2 = weight_decay * \
        tf.add_n([tf.nn.l2_loss(v) for v in train_var_list])
    loss = loss_1 + loss_2

    global_step_pl = tf.placeholder(dtype=tf.int32)
    optimizer_op = get_optimizer_2(global_step_pl, loss)
    extra_update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    print('Building the network completed...........')
    print('')

    print('Number of epochs to train : ' + str(num_epochs))
    print('Batch size : ' + str(batch_size))
    print('Number of train samples : ' + str(num_train_samples))
    print('Number of train batches : ' + str(num_train_batches))
    print('Number of validation samples : ' + str(num_valid_samples))
    print('Number of validation batches : ' + str(num_valid_batches))
    print('')

    print('Training the network.....................')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    ss = tf.Session(config=tf.ConfigProto(device_count={'GPU': 1}))
    ss.run(tf.global_variables_initializer())

    train_loss_per_epoch = list()
    valid_loss_per_epoch = list()

    for epoch in range(num_epochs):
        ti = time.time()
        temp_train_loss_per_epoch = 0
        temp_valid_loss_per_epoch = 0

        for batch_id in range(num_train_batches):
            #_, _, loss_per_batch = ss.run([extra_update_op, optimizer_op, loss], feed_dict = {training_pl : bool(config['training'])})
            _, _, loss_per_batch = ss.run([extra_update_op, optimizer_op, loss], feed_dict={
                                          training_pl: bool(config['training']), global_step_pl: epoch})
            temp_train_loss_per_epoch += loss_per_batch

        for batch_id in range(num_valid_batches):
            loss_per_batch = ss.run(
                loss, feed_dict={training_pl: not(config['training'])})
            temp_valid_loss_per_epoch += loss_per_batch

        ti = time.time() - ti
        train_loss_per_epoch.append(temp_train_loss_per_epoch)
        valid_loss_per_epoch.append(temp_valid_loss_per_epoch)

        print('Epoch : ' + str(epoch + 1) + '/' + str(num_epochs) +
              ', time taken : ' + str(ti) + ' sec.')
        print('Avg. training loss : ' + str(temp_train_loss_per_epoch / num_train_batches) +
              ', Avg. validation loss : ' + str(temp_valid_loss_per_epoch / num_valid_batches))
        print('')

        if (epoch + 1) % config['checkpoint_epoch'] == 0:
            save_model(ss, model_directory, config['model_file'], epoch)

    print('Training the network completed...........')
    print('')

    print('Saving the model.........................')
    save_model(ss, model_directory, config['model_file'], epoch)
    train_loss_per_epoch = np.array(train_loss_per_epoch)
    valid_loss_per_epoch = np.array(valid_loss_per_epoch)

    train_loss_per_epoch = np.true_divide(
        train_loss_per_epoch, num_train_batches)
    valid_loss_per_epoch = np.true_divide(
        valid_loss_per_epoch, num_valid_batches)

    losses_dict = dict()
    losses_dict['train_loss'] = train_loss_per_epoch
    losses_dict['valid_loss'] = valid_loss_per_epoch

    np.save(os.path.join(os.getcwd(), os.path.join(
        model_directory, config['model_metrics'])), (losses_dict))

    print('Saving the model completed...............')
    print('')
    ss.close()


def main():
    batch_train()


if __name__ == '__main__':
    main()
