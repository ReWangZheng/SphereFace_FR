import tensorflow as tf

'''
example1 : hot to use tf.gather_nd()
'''
def example1():
    v1 = tf.constant([
        [1,2,3,4,5],
        [6,7,8,9,10],
        [11,12,13,14,15]
    ])
    slice_1 = tf.constant([[0],[1]])
    slice_2 = tf.constant([[0,1],[1,2]])
    sess = tf.Session()
    print(sess.run(tf.gather_nd(v1,slice_1)))
    print(sess.run(tf.gather_nd(v1,slice_2)))
'''
example2:how to use tf.stack()
'''
def example2():
    v1 = tf.constant([[1,2,3,4,5]])
    v2 = tf.constant([[11,22,33,44,55]])
    sess = tf.Session()
    print(sess.run(tf.stack([v1,v2],axis=2)))
'''
example3:how to use tf.scatter_nd()
'''
def example_3():
    slice_1 = tf.constant([[1],[2],[3],[7],[4]])
    shape = tf.constant([10])
    data = tf.constant([5,4,8,1,4])

    s2 = tf.constant([[1],[2]])
    shape2 = tf.constant([4,4])
    data2 = tf.constant([[1,2,4,4],[3,4,1,2]])


    sess = tf.Session()
    print(sess.run(tf.scatter_nd(slice_1,data,shape)))
    print(sess.run(tf.scatter_nd(s2, data2, shape2)))
if __name__ == '__main__':
    example_3()