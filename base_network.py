import tensorflow as tf


class BaseNetwork(object):

    def __init__(self, device):
        self._device = device
        return

    @property
    def vars(self):
        raise NotImplementedError('please override vars method')

    def sync_from(self, src_net, scope=None):
        src_vars = src_net.vars
        dst_vars = self.vars

        sync_ops = []
        with tf.device(self._device), tf.name_scope(scope, 'BaseNetwork') as scope:
            for (src_var, dst_var) in zip(src_vars, dst_vars):
                sync_ops.append(tf.assign(dst_var, src_var))
            return tf.group(*sync_ops, name=scope)


if __name__ == '__main__':
    net = BaseNetwork('/cpu:0')
    net.vars
