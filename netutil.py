from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def fc_initializer(input_channels, dtype=tf.float32):
    def _initializer(shape, dtype=dtype, partition_info=None):
        d = 1.0 / np.sqrt(input_channels)
        return tf.random_uniform(shape, minval=-d, maxval=d)
    return _initializer


def conv_initializer(kernel_width, kernel_height, input_channels, dtype=tf.float32):
    def _initializer(shape, dtype=dtype, partition_info=None):
        d = 1.0 / np.sqrt(input_channels * kernel_width * kernel_height)
        return tf.random_uniform(shape, minval=-d, maxval=d)
    return _initializer


def fc_variable(shape, name):
    name_w = "W_{0}".format(name)
    name_b = "b_{0}".format(name)

    W = tf.get_variable(name_w, shape, initializer=fc_initializer(shape[0]))
    b = tf.get_variable(name_b, shape[1:], initializer=fc_initializer(shape[0]))

    variable_summaries(W, name_w)
    variable_summaries(b, name_b)
    return W, b


def conv_variable(weight_shape, name, deconv=False):
    name_w = "W_{0}".format(name)
    name_b = "b_{0}".format(name)

    w = weight_shape[0]
    h = weight_shape[1]
    if deconv:
        input_channels = weight_shape[3]
        output_channels = weight_shape[2]
    else:
        input_channels = weight_shape[2]
        output_channels = weight_shape[3]
    bias_shape = [output_channels]

    weight = tf.get_variable(name_w, weight_shape, initializer=conv_initializer(w, h, input_channels))
    bias = tf.get_variable(name_b, bias_shape, initializer=conv_initializer(w, h, input_channels))

    variable_summaries(weight, name_w)
    variable_summaries(bias, name_b)
    return weight, bias


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')


def deconv2d(x, W, input_width, input_height, stride):
    filter_height = W.get_shape()[0].value
    filter_width = W.get_shape()[1].value
    out_channel = W.get_shape()[2].value

    out_height, out_width = get2d_deconv_output_size(
        input_height, input_width, filter_height, filter_width, stride, 'VALID')
    batch_size = tf.shape(x)[0]
    output_shape = tf.stack([batch_size, out_height, out_width, out_channel])
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding='VALID')


def get2d_deconv_output_size(input_height, input_width, filter_height, filter_width, stride, padding_type):
    if padding_type == 'VALID':
        out_height = (input_height - 1) * stride + filter_height
        out_width = (input_width - 1) * stride + filter_width
    elif padding_type == 'SAME':
        out_height = input_height * stride
        out_width = input_width * stride
    return out_height, out_width


def flatten_conv_layer(h_conv):
    h_conv_flat_size = np.prod(h_conv.get_shape().as_list()[1:])
    h_conv_flat = tf.reshape(h_conv, [-1, h_conv_flat_size])
    return h_conv_flat_size, h_conv_flat


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


def lstm_last_relevant(output, length):
    ''' get the last relevant frame of the output of tf.nn.dynamica_rnn()
    '''
    batch_size = tf.shape(output)[0]
    max_length = int(output.get_shape()[1])
    output_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, output_size])
    relevant = tf.gather(flat, index)
    return relevant


def variable_summaries(var, name=None):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        with tf.name_scope(name):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)
    return


