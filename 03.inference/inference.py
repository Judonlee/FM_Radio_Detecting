# -*- coding: utf-8 -*-

"""
@Author : zhudong
@Email  : ynzhudong@163.com
@Time   : 2019/3/29 下午1:59
@File   : train.py
desc:
"""

import os, sys
import numpy as np
import random
from tqdm import tqdm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

from sklearn import manifold

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from net_component import FM_DeepModel
from make_torchdata import myDataSet, TruncatedFromOnlyFB, ToTensor


# -------------Scale and visualize the embedding vectors ----------------
# 将降维后的数据可视化,2维
def plot_embedding_2d(X, y, model_dir, prefix, title=None):
    # 坐标缩放到[0,1]区间

    colors = ['C0', 'C1']
    mark = ['^', 's']

    plt.figure(figsize=(8, 6), dpi=150)

    for idx, c1 in enumerate(np.unique(y)):
        plt.scatter(
            X[y == c1, 0],
            X[y == c1, 1],
            c=colors[idx],
            s=24,
            marker=mark[idx],
        )
    plt.legend(['Speech', 'Non-speech'], loc='upper right', fontsize=14)
    dirname = os.path.join(model_dir, prefix)

    if not os.path.exists(dirname):
        os.mkdir(dirname)

    if title is not None:
        plt.title(title)

    save_name = os.path.join(dirname, 'test_2D.png')
    plt.savefig(save_name, bbox_inches='tight')


# 将降维后的数据可视化,3维
def plot_embedding_3d(X, y, model_dir, prefix, title=None):

    colors = ['C0', 'C1']
    mark = ['^', 's']

    fig = plt.figure(figsize=(8, 6), dpi=150)
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    for idx, c1 in enumerate(np.unique(y)):
        ax.scatter(X[y == c1, 0],
                   X[y == c1, 1],
                   X[y == c1, 2],
                   color=colors[idx],
                   s=24,
                   marker=mark[idx])

    ax.legend(['Speech', 'Non-speech'], loc='upper right', fontsize=14)
    dirname = os.path.join(model_dir, prefix)

    if not os.path.exists(dirname):
        os.mkdir(dirname)

    if title is not None:
        plt.title(title)

    save_name = os.path.join(dirname, 'test_3D.png')
    plt.savefig(save_name, bbox_inches='tight')


# ------------------------------------ Predict ---------------------------------------
def predict(testloader, model, device):

    model.eval()

    correct = 0
    total = 0
    targets, preds, last_embedding = [], [], []

    preds_dic = {}
    preds = []
    with torch.no_grad():
        for batch_index, (datas, labels, filename) in tqdm(enumerate(testloader)):

            datas = datas.to(device)
            last_layer, outputs = model(datas)

            _, predicted = torch.max(outputs.data, 1)

            last_embedding.append(last_layer.cpu().numpy()[0])

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            targets += list(labels.cpu().numpy())
            preds += list(predicted.cpu().numpy())

            freq = filename[0].split('/')[-1].split('M')[0]
            preds_dic[freq] = predicted.cpu().numpy()[0]

            preds += list(predicted.cpu().numpy())

    test_acc = 100. * correct / len(testloader.dataset)

    # # t-SNE
    # print("Computing t-SNE embedding ...")
    # tsne = manifold.TSNE(n_components=3, init='pca', random_state=0)
    #
    # X_tsne = tsne.fit_transform(last_embedding)
    #
    # plot_embedding_2d(X_tsne[:, 0:2], targets, './result', 'test')
    # plot_embedding_3d(X_tsne, targets, './result', 'test')

    return preds, preds_dic, test_acc


if __name__ == '__main__':

    data_path = "label_data_list_lfb.txt"

    with open(data_path, 'r') as f:
        datafile = f.read().splitlines()

    random.seed(2018)
    random.shuffle(datafile)
    data_size = len(datafile)

    print('数据集大小：', data_size)

    te_data = datafile[:int(0.2 * data_size)]
    print('测试数据集大小：', len(te_data))

    te_f = open("label_test_list_lfb.txt", "w")
    for idx, dev_line in enumerate(te_data):
        te_f.write(dev_line + "\n")

    te_f.close()

    te_dir = 'label_test_list_lfb.txt'

    # ------------------------- Data loader --------------------------------
    transform = transforms.Compose([
        TruncatedFromOnlyFB(),
        ToTensor()
    ])

    test_dataset = myDataSet(te_dir, transform=transform)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=1,
                             shuffle=False)

    print('Test data size : {}'.format(len(test_loader.dataset)))

    # ------------------------------- Load model ----------------------------
    # use_cuda = torch.cuda.is_available()
    use_cuda = False

    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(device)
    # model = torch.load('../02.trainModel/model/model_speech_noise_FBank_9.pth')
    model = torch.load('../02.trainModel/model/model_speech_noise_FBank_9.pth', map_location='cpu')

    print(model)

    preds, preds_dic, test_acc = predict(test_loader, model, device)

    # print('\npreds: ', preds)           # 预测的标签， 格式： [0, 0, ..., 1]
    # print('\npreds_dic: ', preds_dic)

    """
    preds_dic 预测的文件及其对应的标签，格式：
    {'weak_noise_2019-01-26_12_50_54_90.9': 0, 
    'weak_noise_2019-01-26_12_12_26_90.9': 0,
    ...,
    'strong_noise_2019-01-04_11_39_48_96': 0}}
    """

    print('\ntest_acc: ', test_acc)    # 准确率 （%）







