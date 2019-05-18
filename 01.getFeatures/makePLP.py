import os
import numpy as np
from python_speech_features import delta
from sidekit.frontend.features import *
import scipy.io.wavfile as wav
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')


def normalize_frames(m, Scale=True):
    if Scale:
        return (m - np.mean(m, axis=0))/(np.std(m, axis=0)+2e-12)
    else:
        return (m - np.mean(m, axis=0))


def extract_plp_sdk(wav_path, feature_path, use_delta=True, use_scale=True):
    """
    取 PLP + 一阶 + 二阶 （13+13+13）
    """
    # print('取 PLP + 一阶 + 二阶 （13+13+13）:')
    if not os.path.exists(feature_path):
        # #这里用wav.read() 读取 *.wav 格式的文件，在用tp的音频提取PLP特征时出现错误，改用librosa.load()解决这个问题。
        # (rate, sig) = wav.read(wav_path)
        
        
        tmp = 'tmp'
        
        if not os.path.exists(tmp):
            os.makedirs(tmp)
        
        tmp_filename = 'tmp/__tmp__.wav'

        os.system('sox {} -c 1 -r 16000 -b 16 {}'.format(wav_path, tmp_filename))

        (rate, sig) = wav.read(tmp_filename)
        
#         import librosa
#         (sig, rate) = librosa.load(wav_path, sr=16000, mono=True)
        # print(rate, sig)

        plp_features = plp(sig, nwin=0.025, fs=16000, plp_order=13, shift=0.01, get_spec=False, get_mspec=False, prefac=0.97, rasta=True)
        plp_features = plp_features[0]

        if use_delta:
            d_plps = delta(plp_features, N=1)
            dd_plps = delta(d_plps, N=1)

            plp_features = normalize_frames(plp_features, Scale=use_scale)
            d_plps = normalize_frames(d_plps, Scale=use_scale)
            dd_plps = normalize_frames(dd_plps, Scale=use_scale)

            frames_feature = np.hstack([plp_features, d_plps, dd_plps])
            # print(frames_feature.shape)
            # use_delta: (409, 13) (409, 39) (nframes,fetures)
#             print('use_delta:', plp_features.shape, frames_feature.shape)
        else:
            plp_features = normalize_frames(plp_features, Scale=use_scale)
            frames_feature = plp_features
            # not use_delta: (409, 13)  (nframes,fetures)
            # print('not use_delta:', frames_feature.shape)
        np.save(feature_path, frames_feature)


if __name__ == '__main__':

    data_dir = "./features/sidekit_PLP/"

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # make data feature
    data_path = "data_list.txt"
    with open(data_path, 'r') as f:
        datafile = f.read().splitlines()
    for line in tqdm(datafile):
        # print(line)
        filename = line.split("/")[-1].split('.wav')[0]

        label = line.split("/")[-3]

        filename = label +'_' + filename + '.npy'
        extract_plp_sdk(os.path.join(line), os.path.join(data_dir, filename),
                        use_delta=True, use_scale=True)
        break
    print('Extract Data PLP Features Done!')
