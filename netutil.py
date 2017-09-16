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
    W = tf.get_variable('W_{0}'.format(name), shape, initializer=fc_initializer(shape[0]))
    b = tf.get_variable('b_{0}'.format(name), shape[1:], initializer=fc_initializer(shape[0]))
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
    output_shape = tf.stack(
        [batch_size, out_height, out_width, out_channel])
    return tf.nn.conv2d_transpose(x, W, output_shape,
                                  strides=[1, stride, stride, 1],
                                  padding='VALID')


def get2d_deconv_output_size(input_height, input_width, filter_height, filter_width, stride, padding_type):
    if padding_type == 'VALID':
        out_height = (input_height - 1) * stride + filter_height
        out_width = (input_width - 1) * stride + filter_width
    elif padding_type == 'SAME':
        out_height = input_height * stride
        out_width = input_width * stride
    return out_height, out_width


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


def lstm_last_relevant(output, length):
    '''
    get the last relevant frame of the output of tf.nn.dynamica_rnn()
    '''
    batch_size = tf.shape(output)[0]
    max_length = int(output.get_shape()[1])
    output_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, output_size])
    relevant = tf.gather(flat, index)
    return relevant


def update_target_graph_op(trainable_vars, tau=0.001):
    '''
    theta_prime = tau * theta + (1 - tau) * theta_prime
    '''
    size = len(trainable_vars)
    update_ops = []
    for i, var in enumerate(trainable_vars[0:size / 2]):
        target = trainable_vars[size // 2 + i]
        # op = tf.assign(target, tau * var.value() + (1 - tau) * target.value())
        op = tf.assign(target, var.value())
        update_ops.append(op)
    return update_ops


def update_target(session, update_ops):
    session.run(update_ops)
    tf_vars = tf.trainable_variables()
    size = len(tf.trainable_variables())
    theta = session.run(tf_vars[0])
    theta_prime = session.run(tf_vars[size // 2])
    assert(theta.all() == theta_prime.all())
    return
