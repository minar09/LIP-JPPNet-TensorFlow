from PIL import Image
import numpy as np
import tensorflow as tf
import os

# colour map for LIP dataset
lip_label_colours = [(0, 0, 0),  # 0=Background
                     (128, 0, 0),  # 1=Hat
                     (255, 0, 0),  # 2=Hair
                     (0, 85, 0),   # 3=Glove
                     (170, 0, 51),  # 4=Sunglasses
                     (255, 85, 0),  # 5=UpperClothes
                     (0, 0, 85),  # 6=Dress
                     (0, 119, 221),  # 7=Coat
                     (85, 85, 0),  # 8=Socks
                     (0, 85, 85),  # 9=Pants
                     (85, 51, 0),  # 10=Jumpsuits
                     (52, 86, 128),  # 11=Scarf
                     (0, 128, 0),  # 12=Skirt
                     (0, 0, 255),  # 13=Face
                     (51, 170, 221),  # 14=LeftArm
                     (0, 255, 255),  # 15=RightArm
                     (85, 255, 170),  # 16=LeftLeg
                     (170, 255, 85),  # 17=RightLeg
                     (255, 255, 0),  # 18=LeftShoe
                     (255, 170, 0)  # 19=RightShoe
                     ]

# colour map for 10k dataset
fashion_label_colours = [(0, 0, 0),  # 0=Background
                         (128, 0, 0),  # hat
                         (255, 0, 0),  # hair
                         (170, 0, 51),  # sunglasses
                         (255, 85, 0),  # upper-clothes
                         (0, 128, 0),  # skirt
                         (0, 85, 85),  # pants
                         (0, 0, 85),  # dress
                         (0, 85, 0),  # belt
                         (255, 255, 0),  # Left-shoe
                         (255, 170, 0),  # Right-shoe
                         (0, 0, 255),  # face
                         (85, 255, 170),  # left-leg
                         (170, 255, 85),  # right-leg
                         (51, 170, 221),  # left-arm
                         (0, 255, 255),  # right-arm
                         (85, 51, 0),  # bag
                         (52, 86, 128)  # scarf
                         ]

# image mean
IMG_MEAN = np.array((104.00698793, 116.66876762,
                     122.67891434), dtype=np.float32)


def decode_labels(mask, num_images=1, num_classes=20):
    """Decode batch of segmentation masks.

    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.

    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    label_colours = []
    if num_classes == 20:
        label_colours = lip_label_colours
    elif num_classes == 18:
        label_colours = fashion_label_colours

    n, h, w, c = mask.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (
        n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
        pixels = img.load()
        for j_, j in enumerate(mask[i, :, :, 0]):
            for k_, k in enumerate(j):
                if k < num_classes:
                    pixels[k_, j_] = label_colours[k]
        outputs[i] = np.array(img)
    return outputs


def prepare_label(input_batch, new_size, one_hot=True, num_classes=20):
    """Resize masks and perform one-hot encoding.

    Args:
      input_batch: input tensor of shape [batch_size H W 1].
      new_size: a tensor with new height and width.

    Returns:
      Outputs a tensor of shape [batch_size h w 21]
      with last dimension comprised of 0's and 1's only.
    """
    with tf.name_scope('label_encode'):
        # as labels are integer numbers, need to use NN interp.
        input_batch = tf.image.resize_nearest_neighbor(input_batch, new_size)
        # reducing the channel dimension.
        input_batch = tf.squeeze(input_batch, squeeze_dims=[3])
        if one_hot:
            input_batch = tf.one_hot(input_batch, depth=num_classes)
    return input_batch


def inv_preprocess(imgs, num_images):
    """Inverse preprocessing of the batch of images.
       Add the mean vector and convert from BGR to RGB.

    Args:
      imgs: batch of input images.
      num_images: number of images to apply the inverse transformations on.

    Returns:
      The batch of the size num_images with the same spatial dimensions as the input.
    """
    n, h, w, c = imgs.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (
        n, num_images)
    outputs = np.zeros((num_images, h, w, c), dtype=np.uint8)
    for i in range(num_images):
        outputs[i] = (imgs[i] + IMG_MEAN)[:, :, ::-1].astype(np.uint8)
    return outputs


def save(saver, sess, logdir, step):
    '''Save weights.   
    Args:
     saver: TensorFlow Saver object.
     sess: TensorFlow session.
     logdir: path to the snapshots directory.
     step: current training step.
    '''
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)
    # saver.save(sess, checkpoint_path)
    print('The checkpoint has been created.')


def load(saver, sess, ckpt_path):
    '''Load trained weights.

    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    '''
    ckpt = tf.train.get_checkpoint_state(ckpt_path)
    if ckpt and ckpt.model_checkpoint_path:
        # ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        # saver.restore(sess, os.path.join(ckpt_path, ckpt_name))
        saver.restore(sess, ckpt.model_checkpoint_path)
        # print("Restored model parameters from {}".format(ckpt_name))
        print("Restored model parameters")
        return True
    else:
        return False
