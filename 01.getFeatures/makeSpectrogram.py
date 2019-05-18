import os
import numpy as np
import librosa
from scipy.signal.windows import hamming
from tqdm import tqdm
import matplotlib.pyplot as plt


def normalize_frames(m, Scale=True):
    if Scale:
        return (m - np.mean(m, axis=0))/(np.std(m, axis=0)+2e-12)
    else:
        return (m - np.mean(m, axis=0))


def extract_stft_lib(wav_path, feature_path, use_scale=True):
    """
    取短时傅里叶变换，取幅度特征并进行归一化
    """
    if not os.path.exists(feature_path):

        # (rate, sig) = wav.read(wav_path)
        (sig, rate) = librosa.load(wav_path, sr=16000, mono=True)

        S = librosa.stft(sig, n_fft=400, hop_length=160, win_length=400, window=hamming)
        magnitude, phase = librosa.magphase(S)

        plt.imshow(magnitude)
        plt.title('magnitude')
        plt.show()

        feature = np.log1p(magnitude)   # log1p() 操作
        feature = feature.transpose()
        feature = normalize_frames(feature, Scale=use_scale)

        np.save(feature_path, feature)

        print('Spectrogram feature:\n', feature)
        print('Spectrogram feature shape:\n', feature.shape)
        print('Spectrogram feature length:\n', len(feature))


if __name__ == '__main__':

    data_dir = "./features/Spectrogram/"

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # make train feature
    data_path = "data_list.txt"
    with open(data_path, 'r') as f:
        datafile = f.read().splitlines()
    for line in tqdm(datafile):

        filename = line.split("/")[-1].split('.wav')[0]

        label = line.split("/")[-3]

        filename = label + '_' + filename + '.npy'
        extract_stft_lib(os.path.join(line), os.path.join(data_dir, filename), use_scale=True)
        break
    print('Extract Data Spectrogram Features Done!')
