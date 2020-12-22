import tensorflow as tf
'''
this class is writed to manage the tensorboard log
'''
class SummaryUitil:
    def __init__(self,scope):
        self.scope = scope
    def scalar(self,name,tensor):
        with tf.name_scope(self.scope):
            tf.summary.scalar(name,tensor)
    def runtime(self):
        pass
class ModelHelper:
    def __init__(self,save_path,sess:tf.Session,step = 10):
        self.sess = sess
        self.saver = tf.train.Saver()
        self.every_step = step
        self.save_path = save_path
        self.latest = tf.train.latest_checkpoint(self.save_path)
        self.log_opt = None
    def reload(self):
        if self.latest is not None:
            self.saver.restore(self.sess,self.latest)
            return int(self.latest.split('-')[-1])
        return 0
    def save(self,cur_step):
        if cur_step==0 or cur_step % self.every_step == 0:
            self.saver.save(self.sess,self.save_path,global_step=cur_step)
