import tensorflow as tf
from layer import *
from loss import *
from dataset import LFWHelper
from util import *
import cv2
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/home/regan/code/deeplearn/SphereFace/MNIST_data/", one_hot=False, reshape=False)
i=0
class SphereFace:
    def __init__(self,is_train=True,num_cls=10,save_path='../models/'):
        self.graph = tf.Graph()
        self.input_size = [28,28]
        self.embedding_dim = 512
        self.batch = 800
        self.target_step = 7000
        with self.graph.as_default():
            self.log = SummaryUitil(scope="SphereFace")
            self.helper = LFWHelper(batch=self.batch)
            self.inputs = tf.placeholder(dtype=tf.float32,
                                         shape=[self.batch, self.input_size[0],
                                                self.input_size[1], 1])
            self.target = tf.placeholder(dtype=tf.int64, shape=[self.batch])

            with tf.variable_scope("sphereFace"):
                # layer 1
                self.w1_conv = weight_variable(shape=[3,3,1,64],
                                               name="conv_w1")
                self.b1_conv = bias_variable(shape=[64],name="bias1")

                self.out_1 = prelu(conv2d(self.inputs,self.w1_conv,stride=[1,2,2,1]) + self.b1_conv)

                # layer 2
                self.w2_conv = weight_variable(shape=[3,3,64,64],
                                                name="conv_w2"
                                               )
                self.b2_conv = bias_variable(shape=[64],name="bias2")

                self.out_2 = prelu(conv2d(self.out_1,self.w2_conv,stride=[1,1,1,1]) + self.b2_conv)

                # layer 3
                self.w3_conv = weight_variable(shape=[3,3,64,128],
                                                name="conv_w3"
                                               )
                self.b3_conv = bias_variable(shape=[128],name="bias3")

                self.out_3 = prelu(conv2d(self.out_2,self.w3_conv,stride=[1,2,2,1]) + self.b3_conv)

                # layer 4
                self.w4_conv = weight_variable(shape=[3,3,128,128],
                                                name="conv_w4",
                                               )
                self.b4_conv = bias_variable(shape=[128],name="bias4")

                self.out_4 = prelu(conv2d(self.out_3,self.w4_conv,stride=[1,1,1,1]) + self.b4_conv)
                # FC
                self.fc,fc_shape = flatten(self.out_4,"fc4")
                self.w5_fc = weight_variable(shape=[fc_shape[1],self.embedding_dim],name="w5_fc",layer_type="fallten")
                self.b5_fc = bias_variable(shape=[self.embedding_dim],name="b5")
                self.embedding = tf.matmul(self.fc,self.w5_fc)+self.b5_fc
                if is_train:
                    self.logits, self.loss = Loss_ASoftmax(self.embedding,self.target,1,num_cls=10,m=2)
                    self.pred = tf.argmax(self.logits,axis=1)
                    global_step = tf.Variable(0, trainable=False)
                    decay_lr = tf.train.exponential_decay(0.01, global_step, 500, 0.9)
                    self.add_step_op = tf.assign_add(global_step, tf.constant(1))
                    self.trainer = tf.train.AdamOptimizer(decay_lr)
                    self.train_opt = self.trainer.minimize(self.loss)
                    self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits,axis=1),self.target),dtype=tf.float32))
            self.sess = tf.Session()
            self.saver = ModelHelper(save_path, self.sess)
    def embed(self,image,resize=False):
        x_in = np.empty(shape=[180,self.input_size[0],self.input_size[1],3])
        for i in range(0,len(image)):
            x_in[i] = cv2.resize(image[i],dsize=(self.input_size[0],self.input_size[1]))
        if resize:
            pass
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            embed = self.sess.run(self.embedding,feed_dict={self.inputs:x_in})
        return embed[:len(image)]
    def train(self):
        step = self.saver.reload()
        with self.graph.as_default():
            # x_train = x_train / 255.
            # x_train, y_train = self.sess.run(data_train)
            self.sess.run(tf.global_variables_initializer())
            init_op, data_train = self.helper.get_iterator()
            self.sess.run(init_op)
            while 1:
                batch_images, batch_labels = mnist.test.next_batch(self.batch)
                self.sess.run([self.train_opt,self.add_step_op], feed_dict={self.inputs: batch_images, self.target: batch_labels})
                if step % 10 == 0:
                    loss_v,acc,p= self.sess.run([self.loss,self.accuracy,self.pred],feed_dict={self.inputs: batch_images, self.target: batch_labels})
                    print("step {} , the loss value {},the accuracy {}".format(step,loss_v,acc))
                # self.saver.save(step)
                step +=1
                if step == self.target_step:
                    break
def test_model():
    face_model = SphereFace()
    face_model.train()
def FR():
    import os
    face1 = '../faces/Horst_Koehler'
    face2 = '../faces/Ian_McKellen'
    face1_img = []
    face_model = SphereFace()
    for filename in os.listdir(face1):
        face1_img.append(plt.imread(os.path.join(face1,filename)))
    face1_emb= face_model.embed(face1_img,resize=True)
    print(face1_emb)

    face2_img = []
    for filename in os.listdir(face2):
        face2_img.append(plt.imread(os.path.join(face2,filename)))
    face2_emb= face_model.embed(face2_img,resize=True)
    print(face2_emb)

if __name__ == '__main__':
    test_model()