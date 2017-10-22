from models.network import NetWork
import tensorflow as tf

class ValueNet(NetWork):
    def setup(self, is_training, num_classes):
        name = 'valuenet/'
        (self.feed('data')
         .conv([5, 5], 64, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'seg_conv1')
         .conv([5, 5], 128, [2, 2], reuse=self.reuse, biased=True, relu=True, name=name + 'seg_conv2')
         .conv([5, 5], 128, [2, 2], reuse=self.reuse, biased=True, relu=True, name=name + 'seg_conv3')
         .conv([5, 5], 384, [2, 2], reuse=self.reuse, biased=True, relu=True, name=name + 'seg_conv4'))
        (self.feed(name + 'seg_conv4')
         .fc(192, reuse=self.reuse, relu=True, name=name + 'fc_1')
         .dropout(keep_prob=0.75, name=name + 'drop4')
         .fc(1, reuse=self.reuse, relu=False, name=name + 'fc_2'))

class FCN(NetWork):
    def setup(self, is_training, num_classes):
        name = 'fcn/'
        (self.feed('data')
         .conv([5, 5], 64, [1, 1], reuse=self.reuse, biased=True, relu=True, name=name + 'image_conv1')
         .conv([5, 5], 128, [2, 2], reuse=self.reuse, biased=True, relu=True, name=name + 'image_conv2')
         .conv([5, 5], 128, [2, 2], reuse=self.reuse, biased=True, relu=True, name=name + 'image_conv3')
         .conv([5, 5], 2, [2, 2], reuse=self.reuse, biased=True, relu=True, name=name + 'image_conv4') )
        (self.feed(name + 'image_conv2')
         .conv([1, 1], 2, [1, 1], reuse=self.reuse, biased=True, relu=False, name=name + 'score_pool4'))
        mid_shape = tf.shape(self.layers[name + 'score_pool4'])
        (self.feed(name + 'image_conv4')
         .resize(mid_shape[1:3], name=name + 'upscore2'))
        (self.feed(name + 'upscore2', name + 'score_pool4')
         .add(name=name + 'fuse_pool4'))
        final_shape = tf.shape(self.layers['data'])-tf.convert_to_tensor([0, 0, 0, 1])
        (self.feed(name + 'fuse_pool4')
        .resize(final_shape[1:3], name=name + 'upscore'))
if __name__ == '__main__':
    image_batch = tf.placeholder(tf.float32, shape=[3,24,24,3])
    seg_batch = tf.placeholder(tf.float32, shape=[3, 24, 24, 5])
    # cap = tf.placeholder(tf.string, shape=[BATCH_SIZE, 1])

    g = FCN({'data': image_batch})
    gg = ValueNet({'data': seg_batch})
    print()
