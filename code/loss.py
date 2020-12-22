import tensorflow as tf
from layer import *
import matplotlib.pylab as plt

def Asoftmax(x_input, y_target, l, num_cls, m=2, name='asoftmax'):
    shape_in = x_input.shape.as_list()
    with tf.variable_scope(name):
        w = weight_variable(shape=[shape_in[-1], num_cls], name='w')
        # caculate thte logisits value of the net
        # if m is zero
        if m == 0:
            bias = bias_variable(shape_in[shape_in[-1]], 'b')
            logist = tf.matmul(x_input, w) + bias
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_target, logist=logist)
            prob = tf.nn.softmax(logist)
            return prob, loss

        w_normalize = tf.nn.l2_normalize(x=w, axis=0)
        x_nomr = tf.norm(x_input, axis=1)
        logist = tf.matmul(x_input, w_normalize)

        indices = tf.stack([tf.constant([i for i in range(0, shape_in[0])], dtype=tf.int64), y_target], axis=1)

        target_logist = tf.gather_nd(logist, indices)

        cos_theta = tf.div(target_logist, x_nomr)
        if m == 1:
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_target, logits=logist))
            prob = tf.nn.softmax(logist)
            return prob, loss
        elif m == 2:
            # this implementation is different from the paper,but the idea are the same
            # the theta more large,the res will be more small
            res = 2 * tf.multiply(tf.sign(cos_theta), tf.square(cos_theta)) - 1
        elif m == 4:
            # this implementation is different from the paper,but the idea are the same
            # the theta more large,the res will be more small
            cos_th2 = tf.square(cos_theta)
            cos_th4 = tf.pow(cos_theta, 4)
            sign0 = tf.sign(cos_theta)
            sign3 = tf.multiply(tf.sign(2 * cos_th2 - 1), sign0)
            sign4 = 2 * sign0 + sign3 - 3
            res = sign3 * (8 * cos_th4 - 8 * cos_th2 + 1) + sign4
        theta_logist = tf.add(logist,
                              tf.scatter_nd(indices,
                                            tf.subtract(tf.multiply(res, x_nomr), target_logist)
                                            , shape=logist.shape))
        res_logist = (1-l)*logist + l * theta_logist
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_target,logits=res_logist))
        prob = tf.nn.softmax(res_logist)
        return prob,loss
