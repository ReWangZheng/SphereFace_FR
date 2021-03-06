from tensorflow._api.v1.data import Dataset,Iterator
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.examples.tutorials.mnist import input_data

def get_datainfo(LFW_dir):
    labels = os.listdir(LFW_dir)
    data_info = []
    data_label = []
    id = 0
    fg = False
    for person_name in labels:
        person_data = os.path.join(LFW_dir,person_name)
        file_name = os.listdir(person_data)
        if len(file_name)<100:
            continue
        for fn in file_name:
            data_info.append(os.path.join(person_data,fn))
            data_label.append(id)
            if len(data_info) == 4000:
                fg = True
                break
        if fg:
            break
        id +=1
    print(id)
    print(len(data_info))
    return (data_info[:1100],data_label[:1100])
class LFWHelper:
    def __init__(self,LFW_dir = '/home/dataset/LFW/',size = (128,128,3),batch = 500):
        self.cls_num = 5749
        self.size = size
        self.FLW_dir = LFW_dir
        self.data_info = get_datainfo(self.FLW_dir)
        self.dataset = Dataset.from_tensor_slices(self.data_info).map(self.load_img).shuffle(800).batch(batch).repeat()
    def load_img(self,fp,label):
        img_file = tf.read_file(fp)
        #decode the binary data to image
        img_decoded = tf.image.decode_jpeg(img_file, channels=self.size[-1])
        img_resize = tf.image.resize(img_decoded,size=[self.size[0],self.size[1]])
        return img_resize,label
    def get_iterator(self):
        data_iter = self.dataset.make_initializable_iterator()
        iter_init = data_iter.make_initializer(self.dataset)
        data = data_iter.get_next()
        return iter_init,data
def test_dataset():
    helper = LFWHelper()
    init_op,data_train = helper.get_iterator()
    with tf.Session() as sess:
        sess.run(init_op)
        x_train,label = sess.run(data_train)
        print(x_train.shape)
        print(label.shape)
if __name__ == '__main__':
    test_dataset()