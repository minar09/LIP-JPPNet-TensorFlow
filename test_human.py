import os
import numpy as np
from PIL import Image
from tqdm import tqdm


def main():
    # train_image_paths, train_label_paths, val_image_paths, val_label_paths = init_path()
    val_image_paths, val_label_paths = init_path()
    #train_hist = compute_hist(train_image_paths, train_label_paths)
    val_hist = compute_hist(val_image_paths, val_label_paths)
    #show_result(train_hist)
    show_result(val_hist)


def init_path():
    # train_image_dir = 'E:/Dataset/LIP/training/images/'
    # train_label_dir = 'E:/Dataset/LIP/training/labels/'
    # val_image_dir = 'E:/Dataset/LIP/validation/images/'
    # val_label_dir = 'E:/Dataset/LIP/validation/labels/'
    
    val_output_dir = 'E:/Dataset/LIP/output/parsing/val/'
    val_id_list = 'E:/Dataset/LIP/list/val_id.txt'
    val_label_dir = 'E:/Dataset/LIP/validation/labels/'

    # train_image_file_names = os.listdir(train_image_dir)
    # train_label_file_names = os.listdir(train_label_dir)
    # val_image_file_names = os.listdir(val_image_dir)
    # val_label_file_names = os.listdir(val_label_dir)

    # with open('E:/Dataset/LIP/list/train_id.txt', 'w') as file:
        # for id in train_image_file_names:
            # file.write("%s\n" % id[:-4])

    # with open('E:/Dataset/LIP/list/val_id.txt', 'w') as file:
        # for id in val_image_file_names:
            # file.write("%s\n" % id[:-4])

    #train_image_paths = []
    #train_label_paths = []
    # val_image_paths = []
    # val_label_paths = []
    
    val_gt_paths = []
    val_pred_paths = []

    # for file_name in tqdm(train_image_file_names):
        # train_image_paths.append(os.path.join(train_image_dir, file_name))

    # for file_name in tqdm(train_label_file_names):
        # train_label_paths.append(os.path.join(train_label_dir, file_name))

    # for file_name in tqdm(val_image_file_names):
        # val_image_paths.append(os.path.join(val_image_dir, file_name))

    # for file_name in tqdm(val_label_file_names):
        # val_label_paths.append(os.path.join(val_label_dir, file_name))

    # with open('E:/Dataset/LIP/list/train.txt', 'w') as file:
        # for id in train_image_paths:
            # file.write("%s\n" % id)

    # with open('E:/Dataset/LIP/list/val.txt', 'w') as file:
        # for id in val_image_paths:
            # file.write("%s\n" % id)
            
    f = open(val_id_list, 'r')
    for line in f:
        val = line.strip("\n")
        val_gt_paths.append(val_label_dir + val + '.png')
        val_pred_paths.append(val_output_dir + val + '.png')

    #return train_image_paths, train_label_paths, val_image_paths, val_label_paths
    return val_pred_paths, val_gt_paths


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)


def compute_hist(images, labels):
    n_cl = 20
    hist = np.zeros((n_cl, n_cl))

    for img_path, label_path in tqdm(zip(images, labels)):
        label = Image.open(label_path)
        label_array = np.array(label, dtype=np.int32)
        image = Image.open(img_path)
        image_array = np.array(image, dtype=np.int32)

        gtsz = label_array.shape
        imgsz = image_array.shape

        if not gtsz == imgsz:
            image = image.resize((gtsz[1], gtsz[0]), Image.ANTIALIAS)
            image_array = np.array(image, dtype=np.int32)

        hist += fast_hist(label_array, image_array, n_cl)

    return hist


def show_result(hist):

    classes = ['background', 'hat', 'hair', 'glove', 'sunglasses', 'upperclothes',
               'dress', 'coat', 'socks', 'pants', 'jumpsuits', 'scarf', 'skirt',
               'face', 'leftArm', 'rightArm', 'leftLeg', 'rightLeg', 'leftShoe',
               'rightShoe']
    # num of correct pixels
    num_cor_pix = np.diag(hist)
    # num of gt pixels
    num_gt_pix = hist.sum(1)
    print('=' * 50)

    # @evaluation 1: overall accuracy
    acc = num_cor_pix.sum() / hist.sum()
    print('>>>', 'overall/pixel accuracy', acc)
    print('-' * 50)

    # @evaluation 2: mean accuracy & per-class accuracy
    print('Accuracy for each class (pixel accuracy):')
    for i in range(20):
        print('%-15s: %f' % (classes[i], num_cor_pix[i] / num_gt_pix[i]))
    acc = num_cor_pix / num_gt_pix
    print('>>>', 'mean accuracy', np.nanmean(acc))
    print('-' * 50)

    # @evaluation 3: mean IU & per-class IU
    print('IoU for each class:')
    union = num_gt_pix + hist.sum(0) - num_cor_pix
    for i in range(20):
        print('%-15s: %f' % (classes[i], num_cor_pix[i] / union[i]))
    iu = num_cor_pix / (num_gt_pix + hist.sum(0) - num_cor_pix)
    print('>>>', 'mean IoU', np.nanmean(iu))
    print('-' * 50)

    # @evaluation 4: frequency weighted IU
    freq = num_gt_pix / hist.sum()
    print('>>>', 'Freq Weighted IoU', (freq[freq > 0] * iu[freq > 0]).sum())
    print('=' * 50)


if __name__ == '__main__':
    main()
