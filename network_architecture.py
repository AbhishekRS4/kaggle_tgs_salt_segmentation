# @author : Abhishek R S

import os
import numpy as np
import h5py
import tensorflow as tf


class UNet:

    # initialize network parameters
    def __init__(self, num_kernels, data_format, training_pl, densenet_path, num_classes=2):
        self._weights_h5 = h5py.File(densenet_path, 'r')
        self._num_kernels = num_kernels
        self._data_format = data_format
        self._padding = 'SAME'
        self._num_classes = num_classes
        self._training = training_pl
        self._conv_kernel_size = [3, 3]
        self._conv_strides = [1, 1]
        self._conv_tr_kernel_size = [2, 2]
        self._conv_tr_strides = [2, 2]
        self._conv_reduction_kernel_size = [1, 1]
        self._conv_dilation_rates = [1, 2, 4]
        self._feature_map_axis = None
        self._avg_pool_axes = None
        self._blocks = [6, 12, 24, 16]
        self._growth_rate = 32
        self._reduction_rate = 0.5
        self._initializer = tf.contrib.layers.xavier_initializer_conv2d()

        if self._data_format == 'channels_first':
            self._encoder_data_format = 'NCHW'
            self._feature_map_axis = 1
            self._pool_kernel = [1, 1, 3, 3]
            self._pool_strides = [1, 1, 2, 2]
            self._avg_pool_axes = [2, 3]
        else:
            self._encoder_data_format = 'NHWC'
            self._feature_map_axis = -1
            self._pool_kernel = [1, 3, 3, 1]
            self._pool_strides = [1, 2, 2, 1]
            self._avg_pool_axes = [1, 2]

    # build resnet encoder
    def densenet121_encoder(self, features):

        # input : RGB format normalized with mean and std
        # x = x / 255.
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]
        # x = (x - mean) / std

        # build the densenet-121 encoder
        # Stage 0
        self.stage0 = self._conv_layer(features, 'conv1_conv')
        self.stage0 = self._get_batchnorm_layer(self.stage0, 'conv1_bn')
        self.stage0 = self._get_relu_activation(self.stage0, name='conv1_relu')
        # 64 x 128 x 128

        # Stage 1
        self.stage1 = tf.nn.max_pool(self.stage0, ksize=self._pool_kernel, strides=self._pool_strides,
                                     padding=self._padding, data_format=self._encoder_data_format, name='pool1')
        # 64 x 64 x 64

        # Stage 2
        self.stage2_dense = self._dense_block(
            self.stage1, self._blocks[0], name='conv2')
        # 256 x 64 x 64
        self.se1 = self._squeeze_excitation_block(
            self.stage2_dense, 256, name='se_block1_')
        # 256 x 64 x 64
        self.stage2_transition = self._transition_block(
            self.se1, self._reduction_rate, name='pool2')
        # 128 x 32 x 32

        # Stage 3
        self.stage3_dense = self._dense_block(
            self.stage2_transition, self._blocks[1], name='conv3')
        # 512 x 32 x 32
        self.se2 = self._squeeze_excitation_block(
            self.stage3_dense, 512, name='se_block2_')
        # 512 x 32 x 32
        self.stage3_transition = self._transition_block(
            self.se2, self._reduction_rate, name='pool3')
        # 256 x 16 x 16

        # Stage 4
        self.stage4_dense = self._dense_block(
            self.stage3_transition, self._blocks[2], name='conv4')
        # 1024 x 16 x 16
        self.se3 = self._squeeze_excitation_block(
            self.stage4_dense, 1024, name='se_block3_')
        # 1024 x 16 x 16
        self.stage4_transition = self._transition_block(
            self.se3, self._reduction_rate, name='pool4')
        # 512 x 8 x 8

        # Stage 5
        self.stage5_dense = self._dense_block(
            self.stage4_transition, self._blocks[3], name='conv5')
        # 1024 x 8 x 8
        self.se4 = self._squeeze_excitation_block(
            self.stage5_dense, 1024, name='se_block4_')
        # 1024 x 8 x 8

    # build a network based on normal residual encoder
    def custom_u_net(self):

        self.elu5_1 = self._normal_conv_block(
            self.se4, self._num_kernels[5], self._conv_reduction_kernel_size, self._padding, 5, 0)
        # 2048 x 8 x 8

        # decoder
        # decoder 1
        self.up1 = self._decoder_block(self.elu5_1, self._num_kernels[4], 1)
        # 1024 x 16 x 16
        self.concat1 = tf.concat(
            [self.up1, self.se3], axis=self._feature_map_axis, name='concat1')
        # 2048 x 16 x 16
        self.elu6_1 = self._normal_conv_block(
            self.concat1, self._num_kernels[4], self._conv_kernel_size, self._padding, 6, 1)
        self.elu6_2 = self._normal_conv_block(
            self.elu6_1, self._num_kernels[4], self._conv_kernel_size, self._padding, 6, 2)
        # 1024 x 16 x 16
        self.se5 = self._squeeze_excitation_block(
            self.elu6_2, 1024, name='se_block5_')
        # 1024 x 16 x 16

        # decoder 2
        self.up2 = self._decoder_block(self.se5, self._num_kernels[3], 2)
        # 512 x 32 x 32
        self.concat2 = tf.concat(
            [self.up2, self.se2], axis=self._feature_map_axis, name='concat2')
        # 1024 x 32 x 32
        self.elu7_1 = self._normal_conv_block(
            self.concat2, self._num_kernels[3], self._conv_kernel_size, self._padding, 7, 1)
        self.elu7_2 = self._normal_conv_block(
            self.elu7_1, self._num_kernels[3], self._conv_kernel_size, self._padding, 7, 2)
        # 512 x 32 x 32
        self.se6 = self._squeeze_excitation_block(
            self.elu7_2, 512, name='se_block6_')
        # 512 x 32 x 32

        # decoder 3
        self.up3 = self._decoder_block(self.se6, self._num_kernels[2], 3)
        # 256 x 64 x 64
        self.concat3 = tf.concat(
            [self.up3, self.se1], axis=self._feature_map_axis, name='concat3')
        # 512 x 64 x 64
        self.elu8_1 = self._normal_conv_block(
            self.concat3, self._num_kernels[2], self._conv_kernel_size, self._padding, 8, 1)
        self.elu8_2 = self._normal_conv_block(
            self.elu8_1, self._num_kernels[2], self._conv_kernel_size, self._padding, 8, 2)
        # 256 x 64 x 64
        self.se7 = self._squeeze_excitation_block(
            self.elu8_2, 256, name='se_block7_')
        # 256 x 64 x 64

        # decoder 4
        self.up4 = self._decoder_block(self.se7, self._num_kernels[0], 4)
        # 64 x 128 x 128
        self.concat4 = tf.concat(
            [self.up4, self.stage0], axis=self._feature_map_axis, name='concat4')
        # 128 x 128 x 128
        self.elu9_1 = self._normal_conv_block(
            self.concat4, self._num_kernels[0], self._conv_kernel_size, self._padding, 9, 1)
        self.elu9_2 = self._normal_conv_block(
            self.elu9_1, self._num_kernels[0], self._conv_kernel_size, self._padding, 9, 2)
        self.se8 = self._squeeze_excitation_block(
            self.elu9_2, 64, name='se_block8_')
        # 64 x 128 x 128

        # decoder 5
        self.logits = self._get_conv2d_layer(
            self.se8, self._num_classes, self._conv_reduction_kernel_size, self._conv_strides, self._padding, name='logits')
        # 2 x 128 x 128

    #---------------------------------------#
    # pretrained densenet encoder functions #
    #---------------------------------------#
    #-----------------------#
    # dense block           #
    #-----------------------#
    def _dense_block(self, input_layer, blocks, name):
        x = input_layer

        for i in range(1, blocks + 1):
            x = self._conv_block(x, self._growth_rate,
                                 name=name + '_block' + str(i))

        return x

    #-----------------------#
    # convolution block     #
    #-----------------------#
    def _conv_block(self, input_layer, growth_rate, name):
        x = self._get_batchnorm_layer(input_layer, name=name + '_0_bn')
        x = self._get_relu_activation(x, name=name + '_0_relu')
        x = self._conv_layer(x, name=name + '_1_conv')

        x = self._get_batchnorm_layer(x, name=name + '_1_bn')
        x = self._get_relu_activation(x, name=name + '_1_relu')
        x = self._conv_layer(x, name=name + '_2_conv')

        x = tf.concat([input_layer, x],
                      axis=self._feature_map_axis, name=name + '_concat')

        return x

    #-----------------------#
    # transition block      #
    #-----------------------#
    def _transition_block(self, input_layer, reduction_rate, name):
        x = self._get_batchnorm_layer(input_layer, name=name + '_bn')
        x = self._get_relu_activation(x, name=name + '_relu')
        x = self._conv_layer(x, name=name + '_conv')

        x = self._avg_pool(x, [2, 2], [2, 2], name=name + '_pool')

        return x

    #-----------------------#
    # avg pool layer        #
    #-----------------------#
    def _avg_pool(self, input_layer, pool_size=[2, 2], strides=[2, 2], name='avg_pool'):
        return tf.layers.average_pooling2d(input_layer, pool_size=pool_size, strides=strides, padding=self._padding, data_format=self._data_format, name=name)

    #-----------------------#
    # convolution2d layer   #
    #-----------------------#
    def _conv_layer(self, input_layer, name, strides=[1, 1, 1, 1]):
        weights_key = name.split('_')

        if len(weights_key) == 4:
            weights_hierarchy = self._weights_h5[weights_key[0]][weights_key[1]][weights_key[2]
                                                                                 ][weights_key[3]][weights_key[0]][weights_key[1]][weights_key[2]][weights_key[3]]
        else:
            weights_hierarchy = self._weights_h5[weights_key[0]
                                                 ][weights_key[1]][weights_key[0]][weights_key[1]]

        W_init_value = np.array(
            weights_hierarchy['kernel:0']).astype(np.float32)
        W = tf.get_variable(
            name=name + '_w', initializer=W_init_value, dtype=tf.float32)
        x = tf.nn.conv2d(input_layer, filter=W, strides=strides,
                         padding=self._padding, data_format=self._encoder_data_format, name=name)

        return x

    #-------------------------------------#
    # traiable densenet decoder functions #
    #-------------------------------------#
    #--------------------------------#
    # build normal convolution block #
    #--------------------------------#
    def _normal_conv_block(self, input_layer, num_kernels, kernel_size, padding, num_1, num_2):
        _conv1 = self._get_conv2d_layer(input_layer, num_kernels, kernel_size,
                                        self._conv_strides, padding, name='conv' + str(num_1) + '_' + str(num_2))
        _bn1 = self._get_batchnorm_layer(
            _conv1, name='bn' + str(num_1) + '_' + str(num_2))
        _elu1 = self._get_elu_activation(
            _bn1, name='elu' + str(num_1) + '_' + str(num_2))

        return _elu1

    #---------------------#
    # build decoder block #
    #---------------------#
    def _decoder_block(self, input_layer, num_kernels, num_1):
        _conv_tr = self._get_conv2d_transpose_layer(
            input_layer, num_kernels, self._conv_tr_kernel_size, self._conv_tr_strides, self._padding, name='up_tr_conv' + str(num_1))
        _bn = self._get_batchnorm_layer(_conv_tr, name='up_bn' + str(num_1))
        _elu = self._get_elu_activation(_bn, name='up_elu' + str(num_1))

        return _elu

    #------------------------------#
    # squeeze and excitation block #
    #------------------------------#
    def _squeeze_excitation_block(self, input_layer, input_channels, ratio=16, name='se_block'):

        _avg_pool = tf.reduce_mean(
            input_layer, axis=self._avg_pool_axes, name=name + 'avg_pool', keepdims=True)
        _conv1 = self._get_conv2d_layer(_avg_pool, int(
            input_channels / ratio), self._conv_reduction_kernel_size, self._conv_strides, self._padding, name=name + 'squeeze')
        _relu = self._get_relu_activation(_conv1, name=name + 'relu')
        _conv2 = self._get_conv2d_layer(_relu, input_channels, self._conv_reduction_kernel_size,
                                        self._conv_strides, self._padding, name=name + 'excitation')
        _sigmoid = self._get_sigmoid_activation(_conv2, name=name + 'sigmoid')
        _scale = tf.multiply(input_layer, _sigmoid, name=name + 'scale')

        return _scale

    #--------------------------------------#
    # common functions used in the network #
    #--------------------------------------#
    #----------#
    # avg pool #
    #----------#
    def _get_avg_pool(self, input_layer, pool_size=[2, 2], strides=[2, 2], name='avg_pool'):
        return tf.layers.average_pooling2d(input_layer, pool_size=pool_size, strides=strides, padding=self._padding, data_format=self._data_format, name=name)

    #---------------------#
    # Convolution2D layer #
    #---------------------#
    def _get_conv2d_layer(self, input_layer, num_filters, kernel_size, strides, padding, name='conv', dilation_rate=[1, 1], use_bias=False):
        return tf.layers.conv2d(inputs=input_layer, filters=num_filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=self._data_format, dilation_rate=dilation_rate, kernel_initializer=self._initializer, use_bias=use_bias, name=name)

    #-------------------------#
    # elu activation function #
    #-------------------------#
    def _get_elu_activation(self, input_layer, name='elu'):
        return tf.nn.elu(input_layer, name=name)

    #--------------------------#
    # relu activation function #
    #--------------------------#
    def _get_relu_activation(self, input_layer, name='relu'):
        return tf.nn.relu(input_layer, name=name)

    #-----------------------------#
    # sigmoid activation function #
    #-----------------------------#
    def _get_sigmoid_activation(self, input_layer, name='sigmoid'):
        return tf.nn.sigmoid(input_layer, name=name)

    #---------------------------#
    # batch normalization layer #
    #---------------------------#
    def _get_batchnorm_layer(self, input_layer, name='bn'):
        return tf.layers.batch_normalization(input_layer, axis=self._feature_map_axis, training=self._training, name=name)

    #-------------------------------------------#
    # transposed convolution2d upsampling layer #
    #-------------------------------------------#
    def _get_conv2d_transpose_layer(self, input_features, num_filters, kernel_size, strides, padding, use_bias=False, name='conv_tr'):
        return tf.layers.conv2d_transpose(inputs=input_features, filters=num_filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=self._data_format, kernel_initializer=self._initializer, use_bias=use_bias, name=name)
