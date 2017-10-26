from __future__ import print_function

import tensorflow as tf
from util.network_util import backup_session, restore_session


def build_session():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
    sess_config = tf.ConfigProto(
        # intra_op_parallelism_threads=NUM_THREADS
        log_device_placement=False,
        allow_soft_placement=True,
        gpu_options=gpu_options
    )
    sess = tf.Session(config=sess_config)
    return sess


def test_save():

    global_t = tf.Variable(0, trainable=False, name="global_t")
    increment_global_t = tf.assign_add(global_t, 1, name="increment_global_t")

    sess = build_session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    restore_session(saver, sess, "tmp_test")

    for _ in range(100):
        sess.run(increment_global_t)

    step = sess.run(global_t)
    print("save step:", step)
    backup_session(saver, sess, "tmp_test", step)
    return


def test_restore():
    global_t = tf.Variable(0, trainable=False, name="global_t")
    # increment_global_t = tf.assign_add(global_t, 1, name="increment_global_t")

    sess = build_session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    restore_session(saver, sess, "tmp_test")
    print(sess.run(global_t))
    return


def main():
    test_save()
    # test_restore()
    return


if __name__ == '__main__':
    main()
