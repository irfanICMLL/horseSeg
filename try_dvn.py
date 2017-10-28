# coding: utf-8
import os

import numpy as np
import tensorflow as tf

from models.generator import ValueNet,FCN
from utils.data_handle import save_weight, load_weight
from utils.image_reader import ImageReader
random_seed=1234
tf.set_random_seed(random_seed)
coord = tf.train.Coordinator()
eps = 1e-8
img_size=[33,33]
random_scale=False
random_mirror=False
random_crop=False
batch_size=1
data_dir='J:\\artical\\Deep Models for Optmising Multivariate Performance Measures\\weizmann_horse_db\\'
with tf.name_scope("create_inputs"):
    reader = ImageReader(
        data_dir,
        img_size,
        random_scale,
        random_mirror,
        random_crop,
        coord)
    image_batch, label_batch = reader.dequeue(batch_size)
    print("Data is ready!")

## load model
image_batch=tf.cast(image_batch,tf.float32)
g_net = FCN({'data': image_batch})
gen_score_map = g_net.get_output()
gen_sample = tf.nn.softmax(gen_score_map)
real_sample =label_batch
tf.constant(0.5)
random_batch = tf.cond()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

global_init = tf.global_variables_initializer()
local_init = tf.local_variables_initializer()
sess.run(global_init)
sess.run(local_init)
threads = tf.train.start_queue_runners(sess, coord)

fk_=sess.run(fk_batch)
coord.request_stop()
coord.join(threads)
print('end....')