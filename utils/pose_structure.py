
import numpy as np


class PoseCreateLayer(object):

    def __init__(self, num_joint, top, bottom):
        self.num_joint_ = num_joint
        self.top = top
        self.bottom = bottom

        if bottom[0].width() != self.num_joint_ * 2:
            print("The bottom width and num of joint should have the same number.")
        top[0].Reshape(bottom[1].num(), self.num_joint_,
                       self.bottom[1].height(), self.bottom[1].width())

    def pose_create(self):
        bottom_data_points = self.bottom[0]
        top_data_points = self.top[0]

        bottom_num = self.bottom[1].num()
        bottom_height = self.bottom[1].height()
        bottom_width = self.bottom[1].width()
        sigma = 1.0  # 1.0

        for idx in range(bottom_num):

            for j in range(self.num_joint_):
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

            bottom_data_points += self.bottom[0].offset(1)
            top_data_points += self.top[0].offset(1)

    @staticmethod
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

    @staticmethod
    def class_to_joint_second(cls_):
        if cls_ == 4:
            return 0
        elif cls_ == 3:
            return 1
        elif cls_ == 2:
            return 2
        else:
            return -1

    @staticmethod
    def class_to_joint_third(cls_):
        if cls_ == 1:
            return 0
        elif cls_ == 2:
            return 1
        else:
            return -1

    def select_joint(self, num_joint_, cls_):
        if num_joint_ == 9:
            return self.class_to_joint_first(cls_)
        elif num_joint_ == 3:
            return self.class_to_joint_second(cls_)
        elif num_joint_ == 2:
            return self.class_to_joint_third(cls_)
        else:
            print("Unexpected num_joint:", num_joint_)


class PoseEvaluateLayer(object):
    bottom_data = bottom[0].cpu_data()
    top_data = top[0].mutable_cpu_data()
    num = bottom[0].num()
    height = bottom[0].height()
    width = bottom[0].width()
    # LOG(INFO) << "pose_len: " << pose_len

  std.vector<int > x_sum_vector[num_joint_]
  std.vector<int > y_sum_vector[num_joint_]
  int cls_, joint_id

  for (i = 0 i < num ++i)    for (h = 0 h < height ++h)      for (w = 0 w < width ++w)        cls_ =  bottom_data[h * width + w]
        joint_id = selectJointFun(num_joint_, cls_)
        if joint_id >= 0 and joint_id < num_joint_:          x_sum_vector[joint_id].push_back(w)
          y_sum_vector[joint_id].push_back(h)


    for (w = 0 w < num_joint_ * 2 ++w)      top_data[w] = 0

    for (n = 0 n < num_joint_ n++)      if x_sum_vector[n].size() > 0 and y_sum_vector[n].size() > 0:        ave_x = std.accumulate(x_sum_vector[n].begin(), x_sum_vector[n].end(), 0.0)
                                      / x_sum_vector[n].size()
        ave_y = std.accumulate(y_sum_vector[n].begin(), y_sum_vector[n].end(), 0.0)
                                      / y_sum_vector[n].size()
        # LOG(INFO) << "ave_x: " << ave_x << "  ave_y:" << ave_y
        top_data[n*2] = int(ave_x)
        top_data[n*2+1] = int(ave_y)
        # LOG(INFO) << "cls: " << n << "  x: " << int(ave_x) << "  y: " << int(ave_y)
      }

    bottom_data += bottom[0].offset(1)
    top_data += top[0].offset(1)
