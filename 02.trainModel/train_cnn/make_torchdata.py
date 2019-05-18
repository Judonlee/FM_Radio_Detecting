# -*- coding: utf-8 -*-

"""
@Author : zhudong
@Email  : ynzhudong@163.com
@Time   : 2019/3/29 下午1:55
@File   : make_torchdata.py
desc:
"""

import numpy as np
import torch
from torch.utils.data import Dataset

# ！每次随机选取的特征段长度
NUM_PREVIOUS_FRAME = 0
NUM_NEXT_FRAME = 297
NUM_FRAMES = NUM_PREVIOUS_FRAME + NUM_NEXT_FRAME


# 输入特征的shape为 (帧数，特征维度)，即 (frames, features)
class TruncatedFromOnlyFB(object):
    def __init__(self, input_per_file=1):
        super(TruncatedFromOnlyFB, self).__init__()
        self.input_per_file = input_per_file

    def __call__(self, frames_features, ):
        num_frames = len(frames_features)
        network_inputs = []
        FEATURE_DIM = frames_features.shape[1]

        import random

        for i in range(self.input_per_file):
            if num_frames <= NUM_FRAMES:
                frames_slice = np.zeros((NUM_FRAMES, FEATURE_DIM)).astype('float64')
                frames_slice[0:frames_features.shape[0]] = frames_features
            else:
                j = random.randrange(NUM_PREVIOUS_FRAME, num_frames - NUM_NEXT_FRAME)
                frames_slice = frames_features[j - NUM_PREVIOUS_FRAME:j + NUM_NEXT_FRAME]

            network_inputs.append(frames_slice)
            # 这里将一个二维的array追加到 network_inputs中，方便后面转为 shape (1, feature, num_frames)
            # print("原始：", np.array(network_inputs).shape)
        return np.array(network_inputs)


class ToTensor(object):
    def __call__(self, feature):
        if isinstance(feature, np.ndarray):
            # handle numpy array
            tmp = torch.FloatTensor(feature.transpose((0, 2, 1)))
            # print("tmp: ", tmp.shape,tmp)
            return tmp


# 训练数据 Loader
class myDataSet(Dataset):
    def __init__(self, filepath, transform=None):
        with open(filepath) as f:
            splited_line = f.read().splitlines()
        datas = []
        for line in splited_line:
            label = int(str(line.split()[1]))
            filepath = line.split()[0]
            datas.append((filepath, label))

        self.datas = datas
        self.transform = transform

    def __getitem__(self, index):
        fn, label = self.datas[index]
        feature = np.load(fn)
        # print(img, img.shape)
        if self.transform is not None:
            feature = self.transform(feature)
        return feature, label

    def __len__(self):
        return len(self.datas)
