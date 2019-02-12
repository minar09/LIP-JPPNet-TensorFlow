from __future__ import print_function
import os
import time
import tensorflow as tf
import numpy as np
import random
from utils import *
from LIP_model import *

# Hide the warning messages about CPU/GPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# parameters setting
N_CLASSES = 20
INPUT_SIZE = (384, 384)
BATCH_SIZE = 4
SHUFFLE = True
RANDOM_SCALE = True
RANDOM_MIRROR = True
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
POWER = 0.9
NUM_IMAGES = 30462
SAVE_PRED_EVERY = NUM_IMAGES // BATCH_SIZE
NUM_EPOCHS = 30
NUM_STEPS = SAVE_PRED_EVERY * NUM_EPOCHS
SHOW_STEP = 10
p_Weight = 1
s_Weight = 1
DATA_DIR = 'D:/Datasets/LIP/training'
LIST_PATH = 'D:/Datasets/LIP/list/train_rev.txt'
DATA_ID_LIST = 'D:/Datasets/LIP/list/train_id.txt'
SNAPSHOT_DIR = './checkpoint/DeepLabV2'
LOG_DIR = './logs/DeepLabV2'


def main():
    random_seed = random.randint(1000, 9999)    # Generate a random number
    tf.set_random_seed(random_seed)    # Set graph-level seed, provide same sequence of random numbers if for a given random number generator

    # Create queue coordinator.
    coord = tf.train.Coordinator()     # Thread coordinator
    h, w = INPUT_SIZE    # Height & Width of input

    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ParsingReader(DATA_DIR, LIST_PATH, DATA_ID_LIST,
                           INPUT_SIZE, RANDOM_SCALE, RANDOM_MIRROR, SHUFFLE, coord, N_CLASSES)    # Input reader, queue of batches of images, labels and heatmaps
        image_batch, label_batch = reader.dequeue(BATCH_SIZE)
        image_batch075 = tf.image.resize_images(
            image_batch, [int(h * 0.75), int(w * 0.75)])    # Generate 0.75 scale of images
        image_batch050 = tf.image.resize_images(
            image_batch, [int(h * 0.5), int(w * 0.5)])    # Generate 0.50 scale of images

    # Define loss and optimisation parameters.
    base_lr = tf.constant(LEARNING_RATE)    # Tensor for base learning rate
    step_ph = tf.placeholder(dtype=tf.float32, shape=())    # Step placeholder tensor
    learning_rate = tf.scalar_mul(
        base_lr, tf.pow((1 - step_ph / NUM_STEPS), POWER))    # Learning rate tensor, set to decay after specific steps
    optim = tf.train.MomentumOptimizer(learning_rate, MOMENTUM)    # Optimizer: SGD + Momentum

    next_image = image_batch    # Get a image from the input batch
    next_image075 = image_batch075    # Get the 0.75 scaled image input
    next_image050 = image_batch050    # Get the 0.50 scaled image input
    next_label = label_batch

    # Create network.
    with tf.variable_scope('', reuse=False):
        net_100 = JPPNetModel(
            {'data': next_image}, is_training=False, n_classes=N_CLASSES)   # Network for input scale 1.0

    with tf.variable_scope('', reuse=True):
        net_075 = JPPNetModel(
            {'data': next_image075}, is_training=False, n_classes=N_CLASSES)  # Network for input scale 0.75

    with tf.variable_scope('', reuse=True):
        net_050 = JPPNetModel(
            {'data': next_image050}, is_training=False, n_classes=N_CLASSES)  # Network for input scale 0.50

    parsing_out1_100 = net_100.layers['fc1_human']    # Parsing output tensor for input scale 1.0
    parsing_out1_075 = net_075.layers['fc1_human']    # Parsing output tensor for input scale 0.75
    parsing_out1_050 = net_050.layers['fc1_human']    # Parsing output tensor for input scale 0.50

    # combine resize (Combine different scales from each refining step)
    parsing_out1 = tf.reduce_mean(tf.stack([parsing_out1_100,
                                            tf.image.resize_images(
                                                parsing_out1_075, tf.shape(parsing_out1_100)[1:3, ]),
                                            tf.image.resize_images(parsing_out1_050, tf.shape(parsing_out1_100)[1:3, ])]), axis=0)

    # Predictions: ignoring all predictions with labels greater or equal than n_classes - flatten prediction
    raw_prediction_p1 = tf.reshape(parsing_out1, [-1, N_CLASSES])
    raw_prediction_p1_100 = tf.reshape(
        parsing_out1_100, [-1, N_CLASSES])
    raw_prediction_p1_075 = tf.reshape(
        parsing_out1_075, [-1, N_CLASSES])
    raw_prediction_p1_050 = tf.reshape(
        parsing_out1_050, [-1, N_CLASSES])

    label_proc = prepare_label(next_label, tf.stack(parsing_out1.get_shape()[
                               1:3]), one_hot=False)  # [batch_size, h, w]
    label_proc075 = prepare_label(next_label, tf.stack(
        parsing_out1_075.get_shape()[1:3]), one_hot=False)
    label_proc050 = prepare_label(next_label, tf.stack(
        parsing_out1_050.get_shape()[1:3]), one_hot=False)

    raw_gt = tf.reshape(label_proc, [-1, ])  # [batch_size, flat]
    raw_gt075 = tf.reshape(label_proc075, [-1, ])
    raw_gt050 = tf.reshape(label_proc050, [-1, ])

    indices = tf.squeeze(
        tf.where(tf.less_equal(raw_gt, N_CLASSES - 1)), 1)
    indices075 = tf.squeeze(
        tf.where(tf.less_equal(raw_gt075, N_CLASSES - 1)), 1)
    indices050 = tf.squeeze(
        tf.where(tf.less_equal(raw_gt050, N_CLASSES - 1)), 1)

    gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
    gt075 = tf.cast(tf.gather(raw_gt075, indices075), tf.int32)
    gt050 = tf.cast(tf.gather(raw_gt050, indices050), tf.int32)

    prediction_p1 = tf.gather(raw_prediction_p1, indices)    # Parsing prediction 1st phase
    prediction_p1_100 = tf.gather(raw_prediction_p1_100, indices)    # Parsing prediction for scale 1.0
    prediction_p1_075 = tf.gather(
        raw_prediction_p1_075, indices075)    # Parsing prediction for scale 0.75
    prediction_p1_050 = tf.gather(
        raw_prediction_p1_050, indices050)    # Parsing prediction for scale 0.50

    # Pixel-wise softmax loss.
    loss_p1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=prediction_p1, labels=gt))    # Parsing prediction loss
    loss_p1_100 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=prediction_p1_100, labels=gt))
    loss_p1_075 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=prediction_p1_075, labels=gt075))
    loss_p1_050 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=prediction_p1_050, labels=gt050))

    loss_parsing = loss_p1 + loss_p1_100 + loss_p1_075 + loss_p1_050

    trainable_variable = tf.trainable_variables()
    grads = optim.compute_gradients(
        loss_parsing, var_list=trainable_variable)    # Compute gradients of current gpu

    tf.add_to_collection('loss_p1', loss_p1)

    # apply the gradients with our optimizers
    train_op = optim.apply_gradients(grads)    # optimizer

    loss_p1_ave = tf.reduce_mean(tf.get_collection('loss_p1'))

    loss_summary_p1 = tf.summary.scalar("loss_p1_ave", loss_p1_ave)

    loss_summary = tf.summary.merge([loss_summary_p1])
    summary_writer = tf.summary.FileWriter(
        LOG_DIR, graph=tf.get_default_graph())    # Graph summary

    # Set up tf session and initialize variables.
    config = tf.ConfigProto(allow_soft_placement=True,    # Operation will be placed on CPU if no GPU implementation
                            log_device_placement=False)    # No verbose log/output with device id
    config.gpu_options.allow_growth = True    # Allow runtime GPU memory growth
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)

    # Saver for storing checkpoints of the model.
    all_saver_var = tf.global_variables()
    # [v for v in all_saver_var if 'pose' not in v.name and 'parsing' not in v.name]
    restore_var = all_saver_var
    saver = tf.train.Saver(var_list=all_saver_var, max_to_keep=50)
    loader = tf.train.Saver(var_list=restore_var)

    if load(loader, sess, SNAPSHOT_DIR):
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # Iterate over training steps.
    for step in range(NUM_STEPS):
        start_time = time.time()
        feed_dict = {step_ph: step}

        # Apply gradients.
        summary, loss_value, _ = sess.run(
            [loss_summary, loss_parsing, train_op], feed_dict=feed_dict)
        summary_writer.add_summary(summary, step)    # Write to summary
        if step % SAVE_PRED_EVERY == 0:
            save(saver, sess, SNAPSHOT_DIR, step)    # Save model

        if step % SHOW_STEP == 0:
            duration = time.time() - start_time    # Calculate step time
            print('step {:d} \t loss = {:.6f}, ({:.3f} sec/step)'.format(step, loss_value, duration))

    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
    main()
