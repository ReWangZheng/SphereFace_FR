import tensorflow as tf
from layer import *
from loss import *
from dataset import LFWHelper
class SphereFace:
    def __init__(self,is_train=True,num_cls=5749):
        self.graph = tf.Graph()
        self.input_size = [24,24]
        self.embedding_dim = 512


        with self.graph.as_default():
            self.helper = LFWHelper()
            self.inputs = tf.placeholder(dtype=tf.float32,
                                         shape=[150, self.input_size[0],
                                                self.input_size[1], 3])
            self.target = tf.placeholder(dtype=tf.int64, shape=[150])

            with tf.variable_scope("sphereFace"):
                # layer 1
                self.w1_conv = weight_variable(shape=[3,3,3,64],
                                               name="conv_w1")
                self.b1_conv = bias_variable(shape=[64],name="bias1")

                self.out_1 = tf.nn.relu(conv2d(self.inputs,self.w1_conv,stride=[1,2,2,1]) + self.b1_conv)

                # layer 2
                self.w2_conv = weight_variable(shape=[3,3,64,128],
                                                name="conv_w2"
                                               )
                self.b2_conv = bias_variable(shape=[128],name="bias2")

                self.out_2 = tf.nn.relu(conv2d(self.out_1,self.w2_conv,stride=[1,2,2,1]) + self.b2_conv)

                # layer 3
                self.w3_conv = weight_variable(shape=[3,3,128,256],
                                                name="conv_w3"
                                               )
                self.b3_conv = bias_variable(shape=[256],name="bias3")

                self.out_3 = tf.nn.relu(conv2d(self.out_2,self.w3_conv,stride=[1,2,2,1]) + self.b3_conv)

                # layer 4
                self.w4_conv = weight_variable(shape=[3,3,256,512],
                                                name="conv_w4"
                                               )
                self.b4_conv = bias_variable(shape=[512],name="bias4")

                self.out_4 = tf.nn.relu(conv2d(self.out_3,self.w4_conv,stride=[1,2,2,1]) + self.b4_conv)

                # FC
                self.fc,fc_shape = flatten(self.out_4,"fc4")
                self.w5_fc = weight_variable(shape=[fc_shape[1],self.embedding_dim],name="w5_fc")
                self.b5_fc = bias_variable(shape=[512],name="b5")
                self.embedding = tf.nn.relu(tf.matmul(self.fc,self.w5_fc)+self.b5_fc)

                if is_train:
                    self.logits, self.loss = Loss_ASoftmax(self.embedding,self.target,0.1,num_cls)
                    self.train_opt = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.loss)
            self.sess = tf.Session()
    def accracy(self,x_test):
        pass
    def train(self):
        step = 0
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            init_op, data_train = self.helper.get_iterator()
            while 1:
                self.sess.run(init_op)
                x_train, y_train = self.sess.run(data_train)
                self.sess.run(self.train_opt, feed_dict={self.inputs: x_train, self.target: y_train})
                if step % 10 == 0:
                    x_train, y_train = self.sess.run(data_train)
                    loss_v = self.sess.run(self.loss,feed_dict={self.inputs: x_train, self.target: y_train})
                    print("step {} , the loss value {}".format(step,loss_v))
                step +=1
def test_model():
    face_model = SphereFace()
    face_model.train()
if __name__ == '__main__':
    test_model()