
import numpy as np
import tensorflow as tf


def mask_to_joints(masks, num_classes=18, num_joints=9):
    masks = tf.squeeze(masks, squeeze_dims=[3])


def pose_create(bottom_input, top_output, num_joint_):
    bottom_data_points = bottom_input[0]
    top_data_points = top_output[0]

    bottom_num = bottom_input[1].num()
    bottom_height = bottom_input[1].height()
    bottom_width = bottom_input[1].width()
    sigma = 1.0  # 1.0

    for idx in range(bottom_num):

        for j in range(num_joint_):
            center_x = int(bottom_data_points[j * 2])
            center_y = int(bottom_data_points[j * 2 + 1])

            for yy in range(bottom_height):

                for xx in range(bottom_width):
                    index = (j * bottom_height + yy) * bottom_width + xx
                    if center_x == 0 and center_y == 0:
                        top_data_points[index] = 0
                    else:
                        gaussian = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (np.power(
                            yy - center_y, 2.0) + np.power(xx - center_x, 2.0)) * np.power(1 / sigma, 2.0))
                        gaussian = 4 * gaussian  # /4
                        top_data_points[index] = gaussian

        bottom_data_points += bottom_input[0].offset(1)
        top_data_points += top_output[0].offset(1)


def pose_evaluate(bottom, top, num_joint_):
    bottom_data = bottom[0].cpu_data()
    top_data = top[0].mutable_cpu_data()
    num = bottom[0].num()
    height = bottom[0].height()
    width = bottom[0].width()

    x_sum_vector = [0] * num_joint_
    y_sum_vector = [0] * num_joint_

    for i in range(num):

        for h in range(height):
            for w in range(width):
                cls_ = bottom_data[h * width + w]
                joint_id = select_joint(cls_)
                if 0 <= joint_id < num_joint_:
                    x_sum_vector[joint_id] = w
                    y_sum_vector[joint_id] = h

        for w in range(num_joint_ * 2):
            top_data[w] = 0

        for n in range(num_joint_):
            if x_sum_vector[n] > 0 and y_sum_vector[n] > 0:
                ave_x = np.mean(x_sum_vector[n])
                ave_y = np.mean(y_sum_vector[n])
                # LOG(INFO) << "ave_x: " << ave_x << "  ave_y:" << ave_y
                top_data[n*2] = int(ave_x)
                top_data[n*2+1] = int(ave_y)
                # LOG(INFO) << "cls: " << n << "  x: " << int(ave_x) << "  y: " << int(ave_y)

        bottom_data += bottom[0].offset(1)
        top_data += top[0].offset(1)


def check_data(bottom, top, num_joint_):
    if bottom[0].num() == bottom[1].num():
        print("The bottom data should have the same number.")
    if bottom[0].channels() == bottom[1].channels():
        print("The bottom data should have the same channel.")
    if bottom[0].height() == bottom[1].height():
        print("The bottom data should have the same height.")
    if bottom[0].width() == bottom[1].width():
        print("The bottom data should have the same width.")
    if bottom[0].width() == num_joint_ * 2:
        print("The bottom data should have the same width as double num_joint_.")
    top[0].Reshape(bottom[0].num(), 1, 1, 1)


def pose_error(bottom, top, num_joint_, error_order_):
    bottom_data_one = bottom[0].cpu_data()
    bottom_data_two = bottom[1].cpu_data()
    bottom_data_three = None
    bottom_data_four = None
    if error_order_ == 2:
        bottom_data_three = bottom[2].cpu_data()
        bottom_data_four = bottom[3].cpu_data()

    top_data = top[0].mutable_cpu_data()
    num = bottom[0].num()
    x1, x2, y1, y2 = 0, 0, 0, 0
    left_arm = 3
    right_arm = 4
    # left_leg = 5, right_leg = 6, left_shoe = 7, right_shoe = 8

    for i in range(num):

        total_distance = 0

        for j in range(num_joint_):
            x1 = bottom_data_one[j*2]
            x2 = bottom_data_two[j*2]
            y1 = bottom_data_one[j*2+1]
            y2 = bottom_data_two[j*2+1]
        total_distance += np.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))

        # LOG(INFO) << "dis of 2: " << total_distance
        if error_order_ == 2:
            x1 = bottom_data_three[left_arm*2]
            x2 = bottom_data_four[left_arm*2]
            y1 = bottom_data_three[left_arm*2+1]
            y2 = bottom_data_four[left_arm*2+1]
            total_distance += np.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))
            x1 = bottom_data_three[right_arm*2]
            x2 = bottom_data_four[right_arm*2]
            y1 = bottom_data_three[right_arm*2+1]
            y2 = bottom_data_four[right_arm*2+1]
            total_distance += np.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))

        # LOG(INFO) << "dis plus 1: " << total_distance
        if error_order_ == 1:
            total_distance /= 10
        elif error_order_ == 2:
            total_distance /= 8
        elif error_order_ == 3:
            total_distance /= 5
        else:
            print("Unexpected error_order: ", error_order_)

        # if total_distance > 10:    #   total_distance = 10
        #
        top_data[0] = total_distance
        # LOG(INFO) << "total_distance: " << total_distance
        bottom_data_one += bottom[0].offset(1)
        bottom_data_two += bottom[1].offset(1)
        top_data += top[0].offset(1)
        if error_order_ == 2:
            bottom_data_three += bottom[2].offset(1)
            bottom_data_four += bottom[3].offset(1)


def class_to_joint_first(cls_):
    if cls_ == 1:
        return 0
    elif cls_ == 2:
        return 0
    elif cls_ == 4:
        return 0
    elif cls_ == 13:
        return 0
    elif cls_ == 5:
        return 1
    elif cls_ == 7:
        return 1
    elif cls_ == 11:
        return 1
    elif cls_ == 9:
        return 2
    elif cls_ == 12:
        return 2
    elif cls_ == 14:
        return 3
    elif cls_ == 15:
        return 4
    elif cls_ == 16:
        return 5
    elif cls_ == 17:
        return 6
    elif cls_ == 18:
        return 7
    elif cls_ == 19:
        return 8
    else:
        return -1


def class_to_joint_second(cls_):
    if cls_ == 4:
        return 0
    elif cls_ == 3:
        return 1
    elif cls_ == 2:
        return 2
    else:
        return -1


def class_to_joint_third(cls_):
    if cls_ == 1:
        return 0
    elif cls_ == 2:
        return 1
    else:
        return -1


def select_joint(num_joint_, cls_):
    if num_joint_ == 9:
        return class_to_joint_first(cls_)
    elif num_joint_ == 3:
        return class_to_joint_second(cls_)
    elif num_joint_ == 2:
        return class_to_joint_third(cls_)
    else:
        print("Unexpected num_joint:", num_joint_)
