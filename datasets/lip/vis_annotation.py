import csv
import sys
import random
import os
from PIL import Image, ImageDraw


def plot_joint(rec, img_folder):
    img_name = os.path.join(img_folder, 'images', rec[0])
    print('Image at: ' + img_name)

    img = Image.open(img_name)
    draw = ImageDraw.Draw(img)
    r = 5
    bombs = [[0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 8],
             [8, 9], [10, 11], [11, 12], [13, 14], [14, 15]]
    colors = [(255, 0, 0), (255, 0, 0),
              (0, 255, 0), (0, 255, 0),
              (0, 0, 255), (0, 0, 255), (0, 0, 255),
              (128, 128, 0), (128, 128, 0),
              (128, 0, 128), (128, 0, 128)]
    r = 5
    for b_id in range(len(bombs)):
        b = bombs[b_id]
        color = colors[b_id]
        x1 = rec[b[0] * 3 + 1]
        y1 = rec[b[0] * 3 + 2]
        v1 = rec[b[0] * 3 + 3]

        x2 = rec[b[1] * 3 + 1]
        y2 = rec[b[1] * 3 + 2]
        v2 = rec[b[1] * 3 + 3]

        if v1 != 'nan' and v2 != 'nan':
            draw.line((int(x1), int(y1), int(x2), int(y2)),
                      fill=color, width=5)
        elif v1 != 'nan':
            draw.ellipse((int(x1) - r, int(y1) - r, int(x1) +
                          r, int(y1) + r), fill=color)
        elif v2 != 'nan':
            draw.ellipse((int(x2) - r, int(y2) - r, int(x2) +
                          r, int(y2) + r), fill=color)

    img.show()


def vis_anno(dataSet):
    csv_path = {
        'train': 'lip_train_set.csv',
        'valid': 'lip_val_set.csv',
        'test': 'lip_test_set.csv'
    }
    img_root = {
        'train': '../LIP_dataset/train_set',
        'valid': '../LIP_dataset/val_set',
        'test': '../LIP_dataset/test_set',
    }
    with open(csv_path[dataSet], 'rb') as f:
        reader = csv.reader(f)
        recs = []
        for row in reader:
            recs.append(row)
        random_id = random.randint(0, len(recs) - 1)
        plot_joint(recs[random_id], img_root[dataSet])


def error():
    print('Error!')
    print('Usage: python vis_annotation.py [train|valid|test]')
    sys.exit()


if __name__ == "__main__":

    dataSet = 'valid'

    if len(sys.argv) == 2:
        dataSet = sys.argv[1].lower()
        print dataSet
        if dataSet not in ['train', 'valid', 'test']:
            error()

    elif len(sys.argv) > 2:
        error()

    vis_anno(dataSet)
