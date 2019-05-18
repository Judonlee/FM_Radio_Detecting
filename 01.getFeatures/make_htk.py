# -*- coding: UTF-8 -*-

import os, sys
sys.path.append('./HTK_Copy')

import numpy as np
from tqdm import tqdm

from HTK_Copy.HTK import HCopy, HTKFile

import warnings
warnings.filterwarnings('ignore')


def normalize_frames(m, Scale=True):
    if Scale:
        return (m - np.mean(m, axis=0))/(np.std(m, axis=0)+2e-12)
    else:
        return (m - np.mean(m, axis=0))


def compute_sdc(mfcc, N=7, d=1, p=3, k=7):

    n_samples = mfcc.shape[0]
    sdc = np.zeros([n_samples, N*k])

    for t in range(n_samples):
        for coeff in range(N):
            for block in range(k):
                c_plus = 0
                c_minus = 0

                if t + block*p + d < n_samples:
                    c_plus = mfcc[t + block*p + d][coeff]

                if t + block*p - d >= 0 and t + block*p - d <n_samples:
                    c_minus = mfcc[t + block*p - d][coeff]
                sdc[t][coeff*k + block] = c_plus - c_minus

    return sdc


def extract_feature(wav_path, feature_path, conf):

    if not os.path.exists(feature_path):

        tmp = 'tmp'
        
        if not os.path.exists(tmp):
            os.makedirs(tmp)
        
        tmp_filename = 'tmp/__tmp__.wav'
        tmp_feature = 'tmp/__tmp__.fe'

        os.system('sox {} -c 1 -r 16000 -b 16 {}'.format(wav_path, tmp_filename))

        output = HCopy(conf, tmp_filename, tmp_feature)

        htk = HTKFile()
        htk.load(tmp_feature)

        data = np.array(htk.data)
        # print('没归一化：', data, data.shape)
        # #
        # plt.figure(figsize=(15, 6))
        # plt.imshow(data.T, aspect='auto', origin='lower', interpolation='none')
        # plt.show()

        sdc = compute_sdc(data)

        sdc = normalize_frames(sdc, Scale=True)

        data = normalize_frames(data, Scale=True)
        data = np.hstack([data, sdc])

        # print('归一化后：', data, data.shape)
        # # #
        # plt.figure(figsize=(15, 6))
        # plt.imshow(data.T, aspect='auto', origin='lower', interpolation='none')
        # plt.show()
        
        np.save(feature_path, data)
        os.remove(tmp_filename)
        os.remove(tmp_feature)


if __name__ == '__main__':

    #     ['config/hcopy_lpc.conf',
    #      'config/hcopy_mfcc.conf',
    #      'config/hcopy_lpcepstra.conf',
    #      'config/hcopy_lprefc.conf',
    #      'config/hcopy_plp.conf',
    #      'config/hcopy_fbank.conf',
    #      'config/hcopy_melspec.conf',
    #      'config/hcopy_lpdelcep.conf']

    # 注意更换特征时，修改以下3个路径
    conf = 'config/hcopy_mfcc.conf'
    print('正在提取:', conf)

    train_fe_dir = "./features/mfcc_SDC/train"

    dev_fe_dir = "./features/mfcc_SDC/dev"

    # # 注意更换特征时，修改以下3个路径
    # conf = 'config/hcopy_fbank.conf'
    # print('正在提取:', conf)
    #
    # train_fe_dir = "./features/FBank40/train"
    #
    # dev_fe_dir = "./features/FBank40/dev"

    # 数据集的路径文件
    train_data_path = "train_data_list.txt"

    dev_data_path = "dev_data_list.txt"

    ################################################################################

    # make train data feature
    with open(train_data_path, 'r') as f:
        datafile = f.read().splitlines()

    for wavpath in tqdm(datafile):

        if not os.path.exists(train_fe_dir):
            os.makedirs(train_fe_dir)

        # print('wavpath: ', wavpath)

        wavname = wavpath.split("/")[-1]
        # print('wavname: ', wavname)

        fe_path = os.path.join(train_fe_dir, wavname).replace('.wav', '.npy')

        # print('for save feature path: ', fe_path)

        extract_feature(wavpath, fe_path, conf)
        # break
    print('Extract Train Features Done!')

    ################################################################################

    # make dev_all data feature
    with open(dev_data_path, 'r') as f:
        datafile = f.read().splitlines()

    for wavpath in tqdm(datafile):

        if not os.path.exists(dev_fe_dir):
            os.makedirs(dev_fe_dir)

        # print('wavpath: ', wavpath)

        wavname = wavpath.split("/")[-1]
        # print('wavname: ', wavname)

        fe_path = os.path.join(dev_fe_dir, wavname).replace('.wav', '.npy')

        # print('for save feature path: ', fe_path)

        extract_feature(wavpath, fe_path, conf)
        # break
    print('Extract Dev_all Features Done!')
