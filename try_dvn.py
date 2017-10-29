# coding: utf-8
import os
import argparse
import os
import time
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
time.sleep(2)

from models.generator import ValueNet, FCN
from utils.data_handle import save_weight, load_weight
from utils.image_reader import ImageReader


def gt_value(pred_labels, gt_labels):
    """Compute the ground truth value of some predicted labels."""
    intersect = np.sum(np.min([pred_labels, gt_labels], axis=0))
    union = np.sum(np.max([pred_labels, gt_labels], axis=0))
    if union == 0:
        if intersect == 0:
            iou = 1
        elif intersect > 0:
            iou = 0
    else:
        iou = intersect / union
    return iou


random_seed = 1234
tf.set_random_seed(random_seed)
coord = tf.train.Coordinator()
eps = 1e-8
img_size = [33, 33]
random_scale = False
random_mirror = False
random_crop = False
batch_size = 2
learning_rate = 0.00001
momentum = 0.9
power = 0.9
num_steps = 3000
restore_from = './weights/dvn/20171029_debug'
g_weight_from = ''
d_weight_from = ''
data_dir = '/data/yy.jiang/yifan.liu/horseSeg-master/data/'
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
image_batch = tf.cast(image_batch, tf.float32)
b = tf.zeros(label_batch.get_shape())
a = tf.ones(label_batch.get_shape())
label_batch_b = tf.where(tf.greater(label_batch, 125), a, b)

real_iou = tf.placeholder(tf.float32, [batch_size, 1])
train_seg = tf.placeholder(tf.float32, [batch_size, 33, 33, 1])
train_seg_new = tf.cast(train_seg,tf.uint8)
train_seg_new=tf.squeeze(train_seg_new,squeeze_dims=[3])
train_seg_new = tf.one_hot(train_seg_new, 2)
g_net = FCN({'data': image_batch})
gen_score_map = g_net.get_output()
gen_sample = tf.nn.softmax(gen_score_map)
real_sample = label_batch_b
dvn_input = tf.concat([train_seg_new, image_batch], 3)
dvn_net = ValueNet({'data': dvn_input})
pre_iou = dvn_net.get_output()
print("The model has been created!!")

## get variables
g_restore_var = [v for v in tf.global_variables() if 'valuenet' not in v.name]
dvn_restore_var = [v for v in tf.global_variables() if 'valuenet' in v.name]
g_var = [v for v in tf.trainable_variables() if 'valuenet' not in v.name]
d_var = [v for v in tf.trainable_variables() if 'valuenet' in v.name]
g_trainable_var = g_var
d_trainable_var = d_var

##set loss
label_list = tf.reshape(label_batch_b, [-1, ])
label_list=tf.cast(label_list,tf.uint8)
label = tf.one_hot(label_list, 2)
logist = tf.reshape(gen_score_map, [-1, 2])
mce_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logist))
dvn_loss = tf.reduce_mean(tf.square(real_iou - pre_iou))
g_loss = mce_loss

g_loss_var, g_loss_op = tf.metrics.mean(g_loss)
dvn_loss_var, dvn_loss_op = tf.metrics.mean(dvn_loss)

## set optimizer
iterstep = tf.placeholder(dtype=tf.float32, shape=[], name='iteration_step')

base_lr = tf.constant(learning_rate, dtype=tf.float32, shape=[])
lr = tf.scalar_mul(base_lr,
                   tf.pow((1 - iterstep / num_steps), power))  # learning rate reduce with the time

train_g_op = tf.train.MomentumOptimizer(learning_rate=lr,
                                        momentum=momentum).minimize(g_loss,
                                                                    var_list=g_trainable_var)
train_d_op = tf.train.MomentumOptimizer(learning_rate=lr/100,
                                        momentum=momentum).minimize(dvn_loss,
                                                                    var_list=d_trainable_var)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

global_init = tf.global_variables_initializer()
local_init = tf.local_variables_initializer()
sess.run(global_init)
sess.run(local_init)
## set saver
saver_all = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=2)
trained_step = 0
if os.path.exists(restore_from + 'checkpoint'):
    trained_step = load_weight(restore_from, saver_all, sess)
elif os.path.exists(g_weight_from + 'checkpoint'):
    saver_g = tf.train.Saver(var_list=g_restore_var, max_to_keep=20)
    load_weight(g_weight_from, saver_g, sess)  # the weight is the completely g model
elif os.path.exists(d_weight_from + 'checkpoint'):
    saver_d = tf.train.Saver(var_list=dvn_restore_var, max_to_keep=20)
    load_weight(g_weight_from, saver_d, sess)

threads = tf.train.start_queue_runners(sess, coord)
# begin training
mce = sess.run(mce_loss)
for step in range(num_steps):
    now_step = int(trained_step) + step if trained_step is not None else step
    feed_dict = {iterstep: step}
    gen_sample_, real_sample_ = sess.run([gen_sample, real_sample], feed_dict)
    iou_gen = np.zeros([batch_size, 1])
    predict = gen_sample_[:, :, :, 1]
    predict[predict > 0.5] = 1
    predict[predict <= 0.5] = 0
    predict=np.expand_dims(predict,3)
    iou_now = np.zeros([batch_size, 1])
    for i in range(batch_size):
        iou_now[i, 0] = gt_value(predict[i,:,:,:], real_sample_[i,:,:,:])

    if np.random.rand() >= 0.7:
        iou_gen = np.ones([batch_size, 1])
        train_seg_ = real_sample_
    else:
        if np.random.rand() >= 0.5:
            init_labels = np.zeros(real_sample_.shape)
            gt_indices = np.random.rand(batch_size,33,33,1) > 0.5
            init_labels[gt_indices] = real_sample_[gt_indices]
            train_seg_ = init_labels
            for i in range(batch_size):
                iou_gen[i, 0] = gt_value(train_seg_, real_sample_)
        else:
            train_seg_ = predict
            iou_gen = iou_now

    feed_dict_dvn = {iterstep: now_step, real_iou: iou_gen, train_seg: train_seg_}
    _, _ = sess.run([train_d_op, train_g_op], feed_dict_dvn)
    if step > 0 and step % 50 == 0:
        save_weight(restore_from, saver_all, sess, now_step)

    if step % 50 == 0 or step == num_steps - 1:
        pre_iou_, dvn_loss_, mce_loss_ = sess.run([pre_iou, dvn_loss, mce_loss], feed_dict_dvn)
        print('step={} d_loss={} g_loss={} real_iou={} pre_iou={}'.format(now_step,
                                                                          dvn_loss_,
                                                                          mce_loss_,
                                                                          iou_now,
                                                                          pre_iou_))

coord.request_stop()
coord.join(threads)
print('end....')
