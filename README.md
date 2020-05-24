## Disclaimer
This is a modified repository from [LIP_JPPNet](https://github.com/Engineering-Course/LIP_JPPNet). Please refer to the original repository for more details.

## Joint Body Parsing & Pose Estimation Network (JPPNet)
Xiaodan Liang, Ke Gong, Xiaohui Shen, and Liang Lin, "Look into Person: Joint Body Parsing & Pose Estimation Network and A New Benchmark", T-PAMI 2018. [LIP_JPPNet](https://github.com/Engineering-Course/LIP_JPPNet)

### Introduction
JPPNet is a state-of-art deep learning method for human parsing and pose estimation built on top of [Tensorflow](http://www.tensorflow.org).
This novel joint human parsing and pose estimation network incorporates the multiscale feature connections and iterative location refinement in an end-to-end framework to investigate efficient context modeling and then enable parsing and pose tasks that are mutually beneficial to each other. This unified framework achieves state-of-the-art performance for both human parsing and pose estimation tasks. 

This distribution provides a publicly available implementation for the key model ingredients reported in [paper](https://arxiv.org/pdf/1804.01984.pdf) which is accepted by T-PAMI 2018.
We simplify the network to solve human parsing by exploring a novel self-supervised structure-sensitive learning approach, which imposes human pose structures into the parsing results without resorting to extra supervision. There is also a public implementation (Caffe) of this self-supervised structure-sensitive JPPNet ([SS-JPPNet](https://github.com/Engineering-Course/LIP_SSL)).

### Look into People (LIP) Dataset
The SSL is trained and evaluated on [LIP dataset](http://www.sysu-hcp.net/lip) for human parsing.  Please check it for more model details. The dataset is also available at [google drive](https://drive.google.com/drive/folders/0BzvH3bSnp3E9ZW9paE9kdkJtM3M?usp=sharing) and [baidu drive](http://pan.baidu.com/s/1nvqmZBN).

### Pre-trained models
There is a release of trained models of JPPNet on LIP dataset at [google drive](https://drive.google.com/open?id=1BFVXgeln-bek8TCbRjN6utPAgRE0LJZg) and [baidu drive](https://pan.baidu.com/s/1hQvg1TMIt0JA0yMfjyzQgQ).

### Training
1. Download LIP dataset or prepare your own data and store in $HOME/datasets.
2. For LIP dataset, we have provided images, parsing labels, lists and the left-right flipping labels (labels_rev) for data augmentation. You need to generate the heatmaps of pose labels. We have provided a script for reference.
3. Run train_JPPNet-s2.py to train the JPPNet with two refinement stages.
4. Use evaluate_pose_JPPNet-s2.py and evaluate_parsing_JPPNet-s2.py to generate the results or evaluate the trained models.
5. Note that the LIPReader class is only suit for labels in LIP for the left-right flipping augmentation. If you want to train on other datasets with different labels, you may have to re-write an image reader class.

### Inference
1. Download the pre-trained model and store in checkpoint directory. Or train from scratch using training script.
2. Prepare the images and store in datasets directory or set your custom path in evaluate_parsing_JPPNet-s2.py.
3. Run evaluate_pose_JPPNet-s2.py for pose estimation and evaluate_parsing_JPPNet-s2.py for human parsing.
4. The results are saved in output folder

### Evaluation
1. To evaluate parsing results, run test_human.py after defining your label and parsing output paths.
2. To apply dense CRF to the parsing outputs, run dense_CRF.py 

## Citation
If you use this code for your research, please cite these papers.
```
@article{liang2018look,
  title={Look into Person: Joint Body Parsing \& Pose Estimation Network and a New Benchmark},
  author={Liang, Xiaodan and Gong, Ke and Shen, Xiaohui and Lin, Liang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2018},
  publisher={IEEE}
}

@InProceedings{Gong_2017_CVPR,
  author = {Gong, Ke and Liang, Xiaodan and Zhang, Dongyu and Shen, Xiaohui and Lin, Liang},
  title = {Look Into Person: Self-Supervised Structure-Sensitive Learning and a New Benchmark for Human Parsing},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {July},
  year = {2017}
}
```
