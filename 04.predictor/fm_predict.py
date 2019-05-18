# -*- coding: utf-8 -*-

"""
@Author : zhudong
@Email  : ynzhudong@163.com
@Time   : 2019/3/29 下午1:59
@File   : train.py
desc:
"""

import os, sys
import shutil
import numpy as np
from python_speech_features import delta, fbank
import scipy.io.wavfile as wav
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from net_component import FM_DeepModel
from make_torchdata import myDataSet, TruncatedFromOnlyFB, ToTensor


def normalize_frames(m, Scale=True):
    if Scale:
        return (m - np.mean(m, axis=0)) / (np.std(m, axis=0) + 2e-12)
    else:
        return (m - np.mean(m, axis=0))


def extract_logfbank_psf(wav_path, feature_path, filter_bank=64,
                         use_delta=True, use_logscale=True, use_scale=True):
    """
    取 logfbanks （64 + 64 + 64）或者不用一阶二阶
    """
    if not os.path.exists(feature_path):

        tmp = 'tmp'

        if not os.path.exists(tmp):
            os.makedirs(tmp)

        tmp_filename = 'tmp/__tmp__.wav'

        os.system('sox {} -c 1 -r 16000 -b 16 {}'.format(wav_path, tmp_filename))

        (rate, sig) = wav.read(tmp_filename)

        fbanks, energies = fbank(sig, samplerate=16000, winlen=0.025, winstep=0.01,
                                 nfilt=filter_bank, nfft=512, lowfreq=0, highfreq=None, preemph=0.97)
        if use_logscale:
            log_fbanks = 20 * np.log10(np.maximum(fbanks, 1e-5))

        if use_delta:
            d_fbanks = delta(log_fbanks, N=1)
            dd_fbanks = delta(d_fbanks, N=1)

            log_fbanks = normalize_frames(log_fbanks, Scale=use_scale)
            d_fbanks = normalize_frames(d_fbanks, Scale=use_scale)
            dd_fbanks = normalize_frames(dd_fbanks, Scale=use_scale)

            frames_feature = np.hstack([log_fbanks, d_fbanks, dd_fbanks])

            # use_delta: (378, 64) (378, 192) (nframes,fetures)
            print('use_delta:', log_fbanks.shape, frames_feature.shape)
        else:
            log_fbanks = normalize_frames(log_fbanks, Scale=use_scale)
            frames_feature = log_fbanks

            # use_delta: (378, 64)  (nframes,fetures)
        #             print('not use_delta:', log_fbanks.shape)
        np.save(feature_path, frames_feature)
        os.remove(tmp_filename)


# ------------------------------------ Predict ---------------------------------------
def predict(testloader, model, device):

    model.eval()

    preds_dic = {}
    preds = []

    with torch.no_grad():
        for batch_index, (datas, filename) in tqdm(enumerate(testloader)):

            datas = datas.to(device)
            last_layer, outputs = model(datas)
            # prob = F.softmax(outputs, dim=1)
            # prob = F.softmax(outputs, dim=1)
            prob = torch.sigmoid(outputs)

            _, predicted = torch.max(outputs.data, 1)

            preds += list(predicted.cpu().numpy())

            freq = filename[0].split('/')[-1].split('M')[0]
            label_prob = [predicted.cpu().numpy()[0], list(prob.cpu().numpy()[0])]
            preds_dic[freq] = label_prob
            # preds_dic[freq] = predicted.cpu().numpy()[0]
            # preds_dic[freq] = list(prob.cpu().numpy()[0])

            preds += list(predicted.cpu().numpy())

    return preds, preds_dic


def Model_predict(fe_dir):

    te_dir = fe_dir

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
    # print(device)
    # #model = torch.load('../02.trainModel/model/model_speech_noise_FBank_9.pth')
    model = torch.load('../02.trainModel/model/model_speech_noise_FBank_9.pth', map_location='cpu')

    print(model)

    preds, preds_dic = predict(test_loader, model, device)

    # print('\npreds: ', preds)  # 预测的标签， 格式： [0, 0, ..., 1]
    # print('\npreds_dic: ', preds_dic)

    """
    preds_dic 预测的文件及其对应的标签，格式：
    {'weak_noise_2019-01-26_12_50_54_90.9': 0, 
    'weak_noise_2019-01-26_12_12_26_90.9': 0,
    ...,
    'strong_noise_2019-01-04_11_39_48_96': 0}}
    """

    return preds_dic


def main(wavpath, datalist, data_dir, features_list):

    os.system("find {} -name '*.wav' | sort > {}".format(wavpath, datalist))
    # os.system("find '../data_fm/' -name '*.wav' | sort >data_list.txt".format(wavpath, datalist))

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # make data feature
    with open(datalist, 'r') as f:
        datafile = f.read().splitlines()

    for line in tqdm(datafile):
        filename = line.split("/")[-1].split('.wav')[0]
        filename = filename + '.npy'

        extract_logfbank_psf(os.path.join(line), os.path.join(data_dir, filename),
                             filter_bank=40, use_delta=False, use_logscale=True, use_scale=True)

    print('Extract Data LogFbank Features Done!\n')

    os.system("find '{}' -name '*.npy' | sort > {}".format(data_dir, features_list))

    result_dic = Model_predict(features_list)

    return result_dic


if __name__ == '__main__':

    wavpath = '../data_fm_424/'
    datalist = 'data_list.txt'

    data_dir = "./features/LogFbank/"

    features_list = 'data_list_lfb.txt'

    result_dic = main(wavpath, datalist, data_dir, features_list)

    print('\nresult_dic: {}\n'.format(result_dic))

    list1 = sorted(result_dic.items(), key=lambda x: x[0])
    for idx, item in enumerate(list1):
        # print(item)
        item = list(item)
        print('频点： {}, 信号类别： {}, 是语音信号的概率值： {}\n'.format(item[0], item[1][0], item[1][1][1]))

        if item[1][1][1] >= 0.9:
            if item[1][0] == 1:
                #
                oldpath = os.path.join('../data_fm_424/radioAudio', str(item[0]) + 'MHz.wav')
                newpatn = os.path.join('../05.S2T_Baidu/speech_signal', str(item[0]) + 'MHz.wav')
                shutil.copy(oldpath, newpatn)
