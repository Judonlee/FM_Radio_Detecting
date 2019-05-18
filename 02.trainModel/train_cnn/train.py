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
import sklearn.metrics as sm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from net_component import FM_DeepModel
from make_torchdata import myDataSet, TruncatedFromOnlyFB, ToTensor


def process(Trainloader, Testloader, model, criterion, optimizer, num_epochs):

    # -------------Scale and visualize the embedding vectors ----------------
    # 将降维后的数据可视化,2维
    def plot_embedding_2d(X, y, epoch, model_dir, prefix, title=None):
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

        save_name = os.path.join(dirname, 'epoch_' + str(epoch) + '_2D.png')
        plt.savefig(save_name, bbox_inches='tight')

    # 将降维后的数据可视化,3维
    def plot_embedding_3d(X, y, epoch, model_dir, prefix, title=None):

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

        save_name = os.path.join(dirname, 'epoch_' + str(epoch) + '_3D.png')
        plt.savefig(save_name, bbox_inches='tight')

    # -------------------------------- train -------------------------------------
    def train(train_loader, epoch):

        model.train()  # train
        print_freq = 10

        pbar = tqdm(enumerate(train_loader))
        for batch_idx, (datas, labels) in pbar:

            datas, labels = datas.to(device), labels.to(device)

            # Forward pass
            _, outputs = model(datas)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % print_freq == 0:
                pbar.set_description(
                    'Train Epoch:{:3d}, Batch:[{:6d}/{:6d} ({:3.0f}%)], Loss: {:.6f}, '.format(
                        epoch, batch_idx + 1, len(train_loader), 100. * (batch_idx + 1) / len(train_loader),
                        loss.item()))

    # --------------------------------- test -------------------------------------
    def test(test_loader, epoch):

        model.eval()  # eval
        correct = 0
        total = 0
        targets, preds, last_embedding = [], [], []

        print(len(test_loader))

        with torch.no_grad():
            for batch_index, (datas, labels) in tqdm(enumerate(test_loader)):
                datas, labels = datas.to(device), labels.to(device)

                last_layer, outputs = model(datas)

                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                targets += list(labels.cpu().numpy())
                preds += list(predicted.cpu().numpy())
                last_embedding.append(last_layer.cpu().numpy()[0])

        test_acc = 100. * correct / len(test_loader.dataset)
        confusion_mtx = sm.confusion_matrix(targets, preds)

        return test_acc, confusion_mtx, np.array(last_embedding), targets

    test_accs, confusion_mtxes = [], []

    for epoch in tqdm(range(1, num_epochs+1)):

        train(Trainloader, epoch)

        test_acc, confusion_mtx, last_embedding, targets = test(Testloader, epoch)

        # t-SNE
        print("Computing t-SNE embedding ...")
        tsne = manifold.TSNE(n_components=3, init='pca', random_state=0)

        X_tsne = tsne.fit_transform(last_embedding)

        plot_embedding_2d(X_tsne[:, 0:2], targets, epoch, '../model', 'test')
        plot_embedding_3d(X_tsne, targets, epoch, '../model', 'test')

        test_accs.append(test_acc)
        confusion_mtxes.append(confusion_mtx)
        print('\rEpoch %d, Best test acc = %2.2f%%' % (epoch, max(test_accs)))

        torch.save(model, '../model/model_speech_noise_FBank_{}.pth'.format(epoch))

    return test_accs, confusion_mtxes


# def main():
#     pass


if __name__ == '__main__':

    data_path = "label_data_list_lfb.txt"

    with open(data_path, 'r') as f:
        datafile = f.read().splitlines()

    # #import the necessary module
    # from sklearn.model_selection import train_test_split

    # #split data set into train and test sets
    # x_train, x_test = train_test_split(datafile, test_size = 0.30, random_state = 2018)

    random.seed(2018)
    random.shuffle(datafile)
    data_size = len(datafile)

    print('数据集大小：', data_size)

    train_data = datafile[int(0.2 * data_size):]
    print('训练数据集大小：', len(train_data))

    tr_f = open("label_train_list_lfb.txt", "w")
    for idx, tr_line in enumerate(train_data):
        #     print(idx, tr_line)
        tr_f.write(tr_line + "\n")

    tr_f.close()

    dev_data = datafile[:int(0.2 * data_size)]
    print('测试数据集大小：', len(dev_data))

    dev_f = open("label_dev_list_lfb.txt", "w")
    for idx, dev_line in enumerate(dev_data):
        #     print(idx, dev_line)
        dev_f.write(dev_line + "\n")

    #     break
    dev_f.close()

    train_dir = 'label_train_list_lfb.txt'
    dev_dir = 'label_dev_list_lfb.txt'

    # ------------------------ Hyper parameters ----------------------------

    num_epochs = 10
    num_classes = 2
    batch_size = 120
    learning_rate = 0.001

    # ------------------------- Data loader --------------------------------
    transform = transforms.Compose([
        TruncatedFromOnlyFB(),
        ToTensor()
    ])

    train_dataset = myDataSet(train_dir, transform=transform)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

    test_dataset = myDataSet(dev_dir, transform=transform)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=1,
                             shuffle=False)

    print('Train data size: {}'.format(len(train_loader.dataset)))
    print('Test data size : {}'.format(len(test_loader.dataset)))

    # ------------------------------- Load model ----------------------------
    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda:0" if use_cuda else "cpu")

    model = FM_DeepModel(num_classes).to(device)

    print(model)

    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    # # 下面的type_size是4，因为我们的参数是float32也就是4B，4个字节
    print('\nModel {} : params: {:4f}M\n'.format(model._get_name(), para * 4 / 1024 / 1024))

    print('使用的设备是：', device)

    # ------------------------------ Loss and optimizer ---------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    process(train_loader, test_loader, model, criterion, optimizer, num_epochs)


