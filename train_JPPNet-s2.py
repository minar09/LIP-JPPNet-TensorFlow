from __future__ import print_function

import time
import random
from utils import *
from LIP_model import *

# Hide the warning messages about CPU/GPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set gpu list
GPU_LIST = [0]  # Here I set CUDA to see one/multiple GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in GPU_LIST])
NUM_GPU = len(GPU_LIST)  # number of GPUs to use

# parameters setting
N_CLASSES = 20
INPUT_SIZE = (384, 384)
BATCH_SIZE = 4
BATCH_ITERATION = BATCH_SIZE // NUM_GPU
SHUFFLE = True
RANDOM_SCALE = True
RANDOM_MIRROR = True
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
POWER = 0.9
NUM_IMAGES = 30462
SAVE_PREDICTION_EVERY = NUM_IMAGES // BATCH_SIZE
NUM_PARSING_EPOCHS = 30
NUM_REFINE_EPOCHS = 30
NUM_PARSING_STEPS = SAVE_PREDICTION_EVERY * NUM_PARSING_EPOCHS
NUM_STEPS = SAVE_PREDICTION_EVERY * (NUM_PARSING_EPOCHS + NUM_REFINE_EPOCHS)
SHOW_STEP = 10
p_Weight = 1
s_Weight = 1
DATA_DIR = 'D:/Datasets/LIP/training'
LIST_PATH = 'D:/Datasets/LIP/list/train_rev.txt'
DATA_ID_LIST = 'D:/Datasets/LIP/list/train_id.txt'
SNAPSHOT_DIR = './checkpoint/JPPNet-s2'
LOG_DIR = './logs/JPPNet-s2'


def main():
    random_seed = random.randint(1000, 9999)    # Generate a random number
    # Set graph-level seed, provide same sequence of random numbers if for a given random number generator
    tf.set_random_seed(random_seed)

    # Create queue coordinator.
    coord = tf.train.Coordinator()     # Thread coordinator
    h, w = INPUT_SIZE    # Height & Width of input

    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = LIPReader(DATA_DIR, LIST_PATH, DATA_ID_LIST,
                           INPUT_SIZE, RANDOM_SCALE, RANDOM_MIRROR, SHUFFLE, coord)    # Input reader, queue of batches of images, labels and heatmaps
        image_batch, label_batch, heatmap_batch = reader.dequeue(BATCH_SIZE)
        image_batch075 = tf.image.resize_images(
            image_batch, [int(h * 0.75), int(w * 0.75)])    # Generate 0.75 scale of images
        image_batch050 = tf.image.resize_images(
            image_batch, [int(h * 0.5), int(w * 0.5)])    # Generate 0.50 scale of images
        heatmap_batch = tf.scalar_mul(
            1.0/255, heatmap_batch)    # Multiply heatmap input

    parsing_tower_grads = []    # Gradients from tower scopes
    combined_tower_grads = []    # Gradients from tower scopes

    # Define loss and optimisation parameters.
    base_lr = tf.constant(LEARNING_RATE)    # Tensor for base learning rate
    # Step placeholder tensor
    step_ph = tf.placeholder(dtype=tf.float32, shape=())
    learning_rate = tf.scalar_mul(
        base_lr, tf.pow((1 - step_ph / NUM_STEPS), POWER))    # Learning rate tensor, set to decay after specific steps
    optim = tf.train.MomentumOptimizer(
        learning_rate, MOMENTUM)    # Optimizer: SGD + Momentum

    reduced_loss = None
    parsing_loss = None

    for i in range(NUM_GPU):    # Iterate among defined GPUs
        with tf.device('/gpu:%d' % i):    # Define specific GPU
            # Set tower scope for the GPU
            with tf.name_scope('Tower_%d' % i) as scope:

                reuse1 = False  # Variable for managing GPUs
                reuse2 = True  # Variable for managing GPUs

                if i == 0:
                    reuse1 = False
                    reuse2 = True
                else:
                    reuse1 = True
                    reuse2 = True

                # Get a image from the input batch
                next_image = image_batch[i *
                                         BATCH_ITERATION:(i + 1) * BATCH_ITERATION, :]
                # Get the 0.75 scaled image input
                next_image075 = image_batch075[i *
                                               BATCH_ITERATION:(i + 1) * BATCH_ITERATION, :]
                # Get the 0.50 scaled image input
                next_image050 = image_batch050[i *
                                               BATCH_ITERATION:(i + 1) * BATCH_ITERATION, :]
                # Get the heatmap for the input image
                next_heatmap = heatmap_batch[i *
                                             BATCH_ITERATION:(i + 1) * BATCH_ITERATION, :]
                next_label = label_batch[i *
                                         BATCH_ITERATION:(i + 1) * BATCH_ITERATION, :]

                # Create network.
                with tf.variable_scope('', reuse=reuse1):
                    net_100 = JPPNetModel(
                        {'data': next_image}, is_training=False, n_classes=N_CLASSES)   # Network for input scale 1.0
                with tf.variable_scope('', reuse=reuse2):
                    net_075 = JPPNetModel(
                        {'data': next_image075}, is_training=False, n_classes=N_CLASSES)  # Network for input scale 0.75
                with tf.variable_scope('', reuse=reuse2):
                    net_050 = JPPNetModel(
                        {'data': next_image050}, is_training=False, n_classes=N_CLASSES)  # Network for input scale 0.50

                # parsing net
                # Parsing net for input scale 1.0
                parsing_fea1_100 = net_100.layers['res5d_branch2b_parsing']
                # Parsing net for input scale 0.75
                parsing_fea1_075 = net_075.layers['res5d_branch2b_parsing']
                # Parsing net for input scale 0.50
                parsing_fea1_050 = net_050.layers['res5d_branch2b_parsing']

                # Parsing output tensor for input scale 1.0
                parsing_out1_100 = net_100.layers['fc1_human']
                # Parsing output tensor for input scale 0.75
                parsing_out1_075 = net_075.layers['fc1_human']
                # Parsing output tensor for input scale 0.50
                parsing_out1_050 = net_050.layers['fc1_human']

                # pose net
                # Pose net for input scale 1.0
                resnet_fea_100 = net_100.layers['res4b22_relu']
                # Pose net for input scale 0.75
                resnet_fea_075 = net_075.layers['res4b22_relu']
                # Pose net for input scale 0.50
                resnet_fea_050 = net_050.layers['res4b22_relu']

                # Pose and parsing output refining for input scale 1.0
                with tf.variable_scope('', reuse=reuse1):
                    pose_out1_100, pose_fea1_100 = pose_net(
                        resnet_fea_100, 'fc1_pose')    # Pose output tensor for input scale 1.0
                    pose_out2_100, pose_fea2_100 = pose_refine(
                        pose_out1_100, parsing_out1_100, pose_fea1_100, name='fc2_pose')    # Pose refined output tensor for input scale 1.0
                    parsing_out2_100, parsing_fea2_100 = parsing_refine(
                        parsing_out1_100, pose_out1_100, parsing_fea1_100, name='fc2_parsing')    # Parsing refined output tensor for input scale 1.0
                    parsing_out3_100, parsing_fea3_100 = parsing_refine(
                        parsing_out2_100, pose_out2_100, parsing_fea2_100, name='fc3_parsing')    # Parsing double-refined output tensor for input scale 1.0
                    pose_out3_100, pose_fea3_100 = pose_refine(
                        pose_out2_100, parsing_out2_100, pose_fea2_100, name='fc3_pose')    # Pose double-refined output tensor for input scale 1.0

                # Pose and parsing output refining for input scale 0.75
                with tf.variable_scope('', reuse=reuse2):
                    pose_out1_075, pose_fea1_075 = pose_net(
                        resnet_fea_075, 'fc1_pose')
                    pose_out2_075, pose_fea2_075 = pose_refine(
                        pose_out1_075, parsing_out1_075, pose_fea1_075, name='fc2_pose')
                    parsing_out2_075, parsing_fea2_075 = parsing_refine(
                        parsing_out1_075, pose_out1_075, parsing_fea1_075, name='fc2_parsing')
                    parsing_out3_075, parsing_fea3_075 = parsing_refine(
                        parsing_out2_075, pose_out2_075, parsing_fea2_075, name='fc3_parsing')
                    pose_out3_075, pose_fea3_075 = pose_refine(
                        pose_out2_075, parsing_out2_075, pose_fea2_075, name='fc3_pose')

                # Pose and parsing output refining for input scale 0.50
                with tf.variable_scope('', reuse=reuse2):
                    pose_out1_050, pose_fea1_050 = pose_net(
                        resnet_fea_050, 'fc1_pose')
                    pose_out2_050, pose_fea2_050 = pose_refine(
                        pose_out1_050, parsing_out1_050, pose_fea1_050, name='fc2_pose')
                    parsing_out2_050, parsing_fea2_050 = parsing_refine(
                        parsing_out1_050, pose_out1_050, parsing_fea1_050, name='fc2_parsing')
                    parsing_out3_050, parsing_fea3_050 = parsing_refine(
                        parsing_out2_050, pose_out2_050, parsing_fea2_050, name='fc3_parsing')
                    pose_out3_050, pose_fea3_050 = pose_refine(
                        pose_out2_050, parsing_out2_050, pose_fea2_050, name='fc3_pose')

                # combine resize (Combine different scales from each refining step)
                parsing_out1 = tf.reduce_mean(tf.stack([parsing_out1_100,
                                                        tf.image.resize_images(
                                                            parsing_out1_075, tf.shape(parsing_out1_100)[1:3, ]),
                                                        tf.image.resize_images(parsing_out1_050, tf.shape(parsing_out1_100)[1:3, ])]), axis=0)
                parsing_out2 = tf.reduce_mean(tf.stack([parsing_out2_100,
                                                        tf.image.resize_images(
                                                            parsing_out2_075, tf.shape(parsing_out2_100)[1:3, ]),
                                                        tf.image.resize_images(parsing_out2_050, tf.shape(parsing_out2_100)[1:3, ])]), axis=0)
                parsing_out3 = tf.reduce_mean(tf.stack([parsing_out3_100,
                                                        tf.image.resize_images(
                                                            parsing_out3_075, tf.shape(parsing_out3_100)[1:3, ]),
                                                        tf.image.resize_images(parsing_out3_050, tf.shape(parsing_out3_100)[1:3, ])]), axis=0)
                pose_out1 = tf.reduce_mean(tf.stack([pose_out1_100,
                                                     tf.image.resize_nearest_neighbor(
                                                         pose_out1_075, tf.shape(pose_out1_100)[1:3, ]),
                                                     tf.image.resize_nearest_neighbor(pose_out1_050, tf.shape(pose_out1_100)[1:3, ])]), axis=0)
                pose_out2 = tf.reduce_mean(tf.stack([pose_out2_100,
                                                     tf.image.resize_nearest_neighbor(
                                                         pose_out2_075, tf.shape(pose_out2_100)[1:3, ]),
                                                     tf.image.resize_nearest_neighbor(pose_out2_050, tf.shape(pose_out2_100)[1:3, ])]), axis=0)
                pose_out3 = tf.reduce_mean(tf.stack([pose_out3_100,
                                                     tf.image.resize_nearest_neighbor(
                                                         pose_out3_075, tf.shape(pose_out3_100)[1:3, ]),
                                                     tf.image.resize_nearest_neighbor(pose_out3_050, tf.shape(pose_out3_100)[1:3, ])]), axis=0)

                # Predictions: ignoring all predictions with labels greater or equal than n_classes - flatten prediction
                raw_prediction_p1 = tf.reshape(parsing_out1, [-1, N_CLASSES])
                raw_prediction_p1_100 = tf.reshape(
                    parsing_out1_100, [-1, N_CLASSES])
                raw_prediction_p1_075 = tf.reshape(
                    parsing_out1_075, [-1, N_CLASSES])
                raw_prediction_p1_050 = tf.reshape(
                    parsing_out1_050, [-1, N_CLASSES])

                raw_prediction_p2 = tf.reshape(parsing_out2, [-1, N_CLASSES])
                raw_prediction_p2_100 = tf.reshape(
                    parsing_out2_100, [-1, N_CLASSES])
                raw_prediction_p2_075 = tf.reshape(
                    parsing_out2_075, [-1, N_CLASSES])
                raw_prediction_p2_050 = tf.reshape(
                    parsing_out2_050, [-1, N_CLASSES])

                raw_prediction_p3 = tf.reshape(parsing_out3, [-1, N_CLASSES])
                raw_prediction_p3_100 = tf.reshape(
                    parsing_out3_100, [-1, N_CLASSES])
                raw_prediction_p3_075 = tf.reshape(
                    parsing_out3_075, [-1, N_CLASSES])
                raw_prediction_p3_050 = tf.reshape(
                    parsing_out3_050, [-1, N_CLASSES])

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

                # Parsing prediction 1st phase
                prediction_p1 = tf.gather(raw_prediction_p1, indices)
                # Parsing prediction for scale 1.0
                prediction_p1_100 = tf.gather(raw_prediction_p1_100, indices)
                prediction_p1_075 = tf.gather(
                    raw_prediction_p1_075, indices075)    # Parsing prediction for scale 0.75
                prediction_p1_050 = tf.gather(
                    raw_prediction_p1_050, indices050)    # Parsing prediction for scale 0.50

                # Parsing prediction 2nd phase
                prediction_p2 = tf.gather(raw_prediction_p2, indices)
                prediction_p2_100 = tf.gather(raw_prediction_p2_100, indices)
                prediction_p2_075 = tf.gather(
                    raw_prediction_p2_075, indices075)
                prediction_p2_050 = tf.gather(
                    raw_prediction_p2_050, indices050)

                # Parsing prediction 3rd phase
                prediction_p3 = tf.gather(raw_prediction_p3, indices)
                prediction_p3_100 = tf.gather(raw_prediction_p3_100, indices)
                prediction_p3_075 = tf.gather(
                    raw_prediction_p3_075, indices075)
                prediction_p3_050 = tf.gather(
                    raw_prediction_p3_050, indices050)

                next_heatmap075 = tf.image.resize_nearest_neighbor(
                    next_heatmap, pose_out1_075.get_shape()[1:3])    # heatmap scale 0.75
                next_heatmap050 = tf.image.resize_nearest_neighbor(
                    next_heatmap, pose_out1_050.get_shape()[1:3])    # heatmap scale 0.50

                # Pixel-wise softmax loss.
                loss_p1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=prediction_p1, labels=gt))    # Parsing prediction loss
                loss_p1_100 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=prediction_p1_100, labels=gt))
                loss_p1_075 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=prediction_p1_075, labels=gt075))
                loss_p1_050 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=prediction_p1_050, labels=gt050))

                loss_p2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=prediction_p2, labels=gt))
                loss_p2_100 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=prediction_p2_100, labels=gt))
                loss_p2_075 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=prediction_p2_075, labels=gt075))
                loss_p2_050 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=prediction_p2_050, labels=gt050))

                loss_p3 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=prediction_p3, labels=gt))
                loss_p3_100 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=prediction_p3_100, labels=gt))
                loss_p3_075 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=prediction_p3_075, labels=gt075))
                loss_p3_050 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=prediction_p3_050, labels=gt050))

                loss_s1 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(
                    tf.square(tf.subtract(next_heatmap, pose_out1)), [1, 2, 3])))    # Pose prediction loss
                loss_s1_100 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(
                    tf.square(tf.subtract(next_heatmap, pose_out1_100)), [1, 2, 3])))
                loss_s1_075 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(
                    tf.square(tf.subtract(next_heatmap075, pose_out1_075)), [1, 2, 3])))
                loss_s1_050 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(
                    tf.square(tf.subtract(next_heatmap050, pose_out1_050)), [1, 2, 3])))

                loss_s2 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(
                    tf.square(tf.subtract(next_heatmap, pose_out2)), [1, 2, 3])))
                loss_s2_100 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(
                    tf.square(tf.subtract(next_heatmap, pose_out2_100)), [1, 2, 3])))
                loss_s2_075 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(
                    tf.square(tf.subtract(next_heatmap075, pose_out2_075)), [1, 2, 3])))
                loss_s2_050 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(
                    tf.square(tf.subtract(next_heatmap050, pose_out2_050)), [1, 2, 3])))

                loss_s3 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(
                    tf.square(tf.subtract(next_heatmap, pose_out3)), [1, 2, 3])))
                loss_s3_100 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(
                    tf.square(tf.subtract(next_heatmap, pose_out3_100)), [1, 2, 3])))
                loss_s3_075 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(
                    tf.square(tf.subtract(next_heatmap075, pose_out3_075)), [1, 2, 3])))
                loss_s3_050 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(
                    tf.square(tf.subtract(next_heatmap050, pose_out3_050)), [1, 2, 3])))

                # Parsing loss
                parsing_loss = loss_p1 + loss_p1_100 + loss_p1_075 + loss_p1_050

                # Combined loss
                loss_parsing = loss_p1 + loss_p1_100 + loss_p1_075 + loss_p1_050 + loss_p2 + loss_p2_100 + \
                    loss_p2_075 + loss_p2_050 + loss_p3 + loss_p3_100 + \
                    loss_p3_075 + loss_p3_050    # Total parsing prediction loss
                loss_pose = loss_s1 + loss_s1_100 + loss_s1_075 + loss_s1_050 + loss_s2 + loss_s2_100 + \
                    loss_s2_075 + loss_s2_050 + loss_s3 + loss_s3_100 + \
                    loss_s3_075 + loss_s3_050    # Total pose prediction loss
                reduced_loss = loss_pose * s_Weight + loss_parsing * p_Weight     # Total loss

                trainable_variable = tf.trainable_variables()
                parsing_grads = optim.compute_gradients(
                    parsing_loss, var_list=trainable_variable)  # Compute gradients of current gpu
                combined_grads = optim.compute_gradients(
                    reduced_loss, var_list=trainable_variable)    # Compute gradients of current gpu

                parsing_tower_grads.append(parsing_grads)    # Append gradients of current gpu
                combined_tower_grads.append(combined_grads)    # Append gradients of current gpu

                tf.add_to_collection('loss_p1', loss_p1)
                tf.add_to_collection('loss_p2', loss_p2)
                tf.add_to_collection('loss_p3', loss_p3)
                tf.add_to_collection('loss_s1', loss_s1)
                tf.add_to_collection('loss_s2', loss_s2)
                tf.add_to_collection('loss_s3', loss_s3)
                tf.add_to_collection('parsing_loss', parsing_loss)
                tf.add_to_collection('reduced_loss', reduced_loss)

    # Average the gradients
    parsing_grads_ave = average_gradients(parsing_tower_grads)     # average gradients
    combined_grads_ave = average_gradients(combined_tower_grads)     # average gradients

    # apply the gradients with our optimizers
    parsing_train_op = optim.apply_gradients(parsing_grads_ave)    # optimizer
    combined_train_op = optim.apply_gradients(combined_grads_ave)    # optimizer

    loss_p1_ave = tf.reduce_mean(tf.get_collection('loss_p1'))
    loss_p2_ave = tf.reduce_mean(tf.get_collection('loss_p2'))
    loss_p3_ave = tf.reduce_mean(tf.get_collection('loss_p3'))
    loss_s1_ave = tf.reduce_mean(tf.get_collection('loss_s1'))
    loss_s2_ave = tf.reduce_mean(tf.get_collection('loss_s2'))
    loss_s3_ave = tf.reduce_mean(tf.get_collection('loss_s3'))
    parsing_loss_ave = tf.reduce_mean(tf.get_collection('parsing_loss'))
    loss_ave = tf.reduce_mean(tf.get_collection('reduced_loss'))

    loss_summary_p1 = tf.summary.scalar("loss_p1_ave", loss_p1_ave)
    loss_summary_p2 = tf.summary.scalar("loss_p2_ave", loss_p2_ave)
    loss_summary_p3 = tf.summary.scalar("loss_p3_ave", loss_p3_ave)
    loss_summary_s1 = tf.summary.scalar("loss_s1_ave", loss_s1_ave)
    loss_summary_s2 = tf.summary.scalar("loss_s2_ave", loss_s2_ave)
    loss_summary_s3 = tf.summary.scalar("loss_s3_ave", loss_s3_ave)
    parsing_loss_summary_ave = tf.summary.scalar("parsing_loss_ave", parsing_loss_ave)
    loss_summary_ave = tf.summary.scalar("loss_ave", loss_ave)
    loss_summary = tf.summary.merge([parsing_loss_summary_ave, loss_summary_ave, loss_summary_s1, loss_summary_s2,
                                     loss_summary_s3, loss_summary_p1, loss_summary_p2, loss_summary_p3])
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
    saver = tf.train.Saver(var_list=all_saver_var, max_to_keep=10)
    loader = tf.train.Saver(var_list=restore_var)

    if load(loader, sess, SNAPSHOT_DIR):
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # Iterate over training steps. #
    for step in range(NUM_STEPS):
        start_time = time.time()
        feed_dict = {step_ph: step}

        # Apply gradients.
        if step < NUM_PARSING_STEPS:
            step_name = "Parsing"
            summary, loss_value, _ = sess.run(
                [loss_summary, parsing_loss, parsing_train_op], feed_dict=feed_dict)
        else:
            step_name = "Refining"
            summary, loss_value, _ = sess.run(
                [loss_summary, reduced_loss, combined_train_op], feed_dict=feed_dict)

        summary_writer.add_summary(summary, step)    # Write to summary
        if step % SAVE_PREDICTION_EVERY == 0 and step > 0:
            save(saver, sess, SNAPSHOT_DIR, step)    # Save model

        if step % SHOW_STEP == 0:
            duration = time.time() - start_time    # Calculate step time
            print(step_name,
                  ' step {:d} \t loss = {:.6f}, ({:.3f} sec/step)'.format(step, loss_value, duration))

    coord.request_stop()
    coord.join(threads)


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


if __name__ == '__main__':
    main()
