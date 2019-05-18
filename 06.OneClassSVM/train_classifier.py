# -*- coding: utf-8 -*-

"""
@Author : zhudong
@Email  : ynzhudong@163.com
@Time   : 2019/5/4 上午10:36
@File   : train_classifier.py
desc:
"""
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
import scipy.io.wavfile as wav
from utility import *
from proj_paths import *
from feature_extraction import *
from sklearn.preprocessing import StandardScaler

import os
import time
from tqdm import tqdm

from python_speech_features import sigproc, mfcc, delta
from settings import *
import progressbar

import matplotlib.pyplot as plt


def pca_fit(x):
    pca_obj.fit(np.array(x))


def pca_transform(x):
    pca_x = pca_obj.transform(np.array(x))
    return pca_x


def classify(x):
    # clf = OneClassSVM(nu=0.15, kernel="rbf",
    #                 gamma=0.1)
    clf = OneClassSVM(kernel='rbf', degree=3, gamma=0.1,
                      coef0=0.0, tol=1e-10, nu=0.001, shrinking=True, cache_size=500,
                      verbose=True, max_iter=-1, random_state=None)
    clf.fit(np.array(x))
    return clf


def cal_threshold():
    scores = 0
    # remove previous files

    # for raw_file_name, joined_file_path in collect_files(SVM_DATA_SET_PATH):
    #     print(joined_file_path)
    #     rate, sig = wav.read(joined_file_path)
    #     feat = extract(sig)
    #     pca_feats = pca_transform([feat])
    #
    #     score = clf.score_samples(pca_feats)
    #     scores += score[0]
    #     print(raw_file_name, score)
    #
    #     # os.remove(joined_file_path)

    for raw_file_name, joined_file_path in collect_files(SVM_DATA_SET_PATH):
        print(joined_file_path)
        rate, sig = wav.read(joined_file_path)
        feat = extract(sig)

        score = clf.score_samples(np.array([feat]))
        scores += score[0]
        print(raw_file_name, score)

        # os.remove(joined_file_path)


if __name__ == "__main__":

    # bar = progressbar.ProgressBar(maxval=len(collect_files(SVM_DATA_SET_PATH)),
    #                               widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    # bar.start()
    # print("Extracting features ...")
    # features = []
    #
    # for i, (raw_file_name, joined_file_path) in tqdm(enumerate(collect_files(SVM_DATA_SET_PATH))):
    #     _, sig = wav.read(joined_file_path)
    #     feats = extract(sig)
    #     features.append(feats)
    #     bar.update(i + 1)
    #
    # bar.finish()
    # print("Done.")
    #
    # save(features, SVM_FEATURES_NAME, ".")

    # reading the data from saved models in train
    features = load(SVM_FEATURES_NAME)

    #  3）特征工程，标准化
    transfer = StandardScaler()

    features = transfer.fit_transform(np.array(features))

    #
    pca_obj = PCA(n_components=60, whiten=True)
    pca_fit(features)
    # print(features)
    pca_feats = pca_transform(features)

    clf = classify(pca_feats)

    print('clf.get_params():\n', clf.get_params())

    scores = []
    y_preds = []
    # remove previous files
    for raw_file_name, joined_file_path in tqdm(collect_files(SVM_DATA_SET_PATH)):
        rate, sig = wav.read(joined_file_path)
        feat = extract(sig)
        # print(feat)
        # print(max(feat), min(feat))
        # print(np.array([feat]))

        feat = transfer.transform(np.array([feat]))
        # print(feat)
        pca_feats = pca_transform(feat)

        y_pred_train = clf.predict(pca_feats)

        score = clf.score_samples(pca_feats)
        print(raw_file_name, y_pred_train, score)

        scores.append(score[0])
        y_preds.append(y_pred_train[0])

    itemindex_is = np.argwhere(np.array(y_preds) == -1)
    itemindex_notis = np.argwhere(np.array(y_preds) != -1)

    print('number correct', len(itemindex_is))
    print('number not correct', len(itemindex_notis))

    plt.figure(figsize=(18, 10))
    plt.hist(np.array(scores)[itemindex_is], bins=60, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.hist(np.array(scores)[itemindex_notis], bins=60, normed=0, facecolor="red", edgecolor="black", alpha=0.7)

    # 显示图标题
    # plt.title("频数/频率分布直方图")
    plt.show()
