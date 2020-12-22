import tensorflow as tf
from layer import *
import matplotlib.pylab as plt
def Loss_ASoftmax(x, y, l, num_cls, m = 2, name = 'asoftmax'):
    '''
    x: B x D - data
    y: B x 1 - label
    l: 1 - lambda
    '''
    #compute the output of last laye
    xs = x.get_shape()

    # get the weight of this layer
    # w = tf.get_variable("asoftmax/W", [xs[1], num_cls], dtype=tf.float32,
    #         initializer=tf.contrib.layers.xavier_initializer())
    w = weight_variable([xs[1], num_cls],name=name)
    # esp is set to prevent that the beichushu is zero in div procsing
    eps = 1e-8


    xw = tf.matmul(x,w)

    if m == 0:
        return xw, tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=xw))

    w_norm = tf.norm(w, axis = 0) + eps
    logits = xw/w_norm

    if y is None:
        return logits, None

    ordinal = tf.constant(list(range(0, xs[0])), tf.int64)
    ordinal_y = tf.stack([ordinal, y], axis = 1)

    x_norm = tf.norm(x, axis = 1) + eps

    sel_logits = tf.gather_nd(logits, ordinal_y)

    cos_th = tf.div(sel_logits, x_norm)

    if m == 1:

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))

    else:

        if m == 2:

            cos_sign = tf.sign(cos_th)
            res = 2*tf.multiply(tf.sign(cos_th), tf.square(cos_th)) - 1 # cos2\theta

        elif m == 4:
            cos_th2 = tf.square(cos_th)
            cos_th4 = tf.pow(cos_th, 4)
            sign0 = tf.sign(cos_th)
            sign3 = tf.multiply(tf.sign(2*cos_th2 - 1), sign0)
            sign4 = 2*sign0 + sign3 - 3
            res = sign3*(8*cos_th4 - 8*cos_th2 + 1) + sign4
        else:
            raise ValueError('unsupported value of m')

        scaled_logits = tf.multiply(res, x_norm)

        f = 1.0/(1.0+l)
        ff = 1.0 - f
        comb_logits_diff = tf.add(logits, tf.scatter_nd(ordinal_y, tf.subtract(scaled_logits, sel_logits), logits.get_shape()))
        updated_logits = ff*logits + f*comb_logits_diff

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=updated_logits))
    return logits, loss


def Angular_Softmax_Loss(embeddings, labels, numcls,margin=4):
    """
    Note:(about the value of margin)
    as for binary-class case, the minimal value of margin is 2+sqrt(3)
    as for multi-class  case, the minimal value of margin is 3

    the value of margin proposed by the author of paper is 4.
    here the margin value is 4.
    """
    l = 0
    embeddings_norm = tf.norm(embeddings, axis=1) + 1e-8

    with tf.variable_scope("softmax"):
        weights = tf.get_variable(name='embedding_weights',
                                  shape=[embeddings.get_shape().as_list()[-1], numcls],
                                  initializer=tf.contrib.layers.xavier_initializer())
        weights = tf.nn.l2_normalize(weights, axis=0)
        # cacualting the cos value of angles between embeddings and weights
        orgina_logits = tf.matmul(embeddings, weights)
        N = embeddings.get_shape()[0] # get batch_size
        single_sample_label_index = tf.stack([tf.constant(list(range(N)), tf.int64), labels], axis=1)
        # N = 128, labels = [1,0,...,9]
        # single_sample_label_index:
        # [ [0,1],
        #   [1,0],
        #   ....
        #   [128,9]]
        selected_logits = tf.gather_nd(orgina_logits, single_sample_label_index)
        cos_theta = tf.div(selected_logits, embeddings_norm)
        cos_theta_power = tf.square(cos_theta) #cos^2
        cos_theta_biq = tf.pow(cos_theta, 4) # cos^4
        sign0 = tf.sign(cos_theta)# 1 or -1
        sign3 = tf.multiply(tf.sign(2*cos_theta_power-1), sign0) #cos2^ * sign0 (1 or -1)
        sign4 = 2*sign0 + sign3 -3
        result=sign3*(8*cos_theta_biq-8*cos_theta_power+1) + sign4
        #(1 or -1 )
        margin_logits = tf.multiply(result, embeddings_norm)
        f = 1.0/(1.0+l)
        ff = 1.0 - f
        combined_logits = tf.add(orgina_logits, tf.scatter_nd(single_sample_label_index,
                                                       tf.subtract(margin_logits, selected_logits),
                                                       orgina_logits.get_shape()))
        updated_logits = ff*orgina_logits + f*combined_logits
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=updated_logits))
        pred_prob = tf.nn.softmax(logits=updated_logits)
        return pred_prob, loss
def test_predict_48_24_12():
    import cv2
    from mtcnn.mtcnn import MTCNN
    mtc = MTCNN()
    # 获取摄像头
    capture = cv2.VideoCapture(0)
    capture.set(3, 480)
    while capture.isOpened():
        # 摄像头打开，读取图像
        flag, image = capture.read()
        face_mes= mtc.detect_faces(image)
        if len(face_mes) > 0:
            x,y,w,h = face_mes[0]['box']
            image = cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255))
        cv2.imshow("image", image)
        k = cv2.waitKey(1)
        if k == ord('s'):
            cv2.imwrite("test.jpg", image)
        elif k == ord("q"):
            break
    # 释放摄像头
    capture.release()
    # 关闭所有窗口
    cv2.destroyAllWindows()

if __name__ == '__main__':
    test_predict_48_24_12()