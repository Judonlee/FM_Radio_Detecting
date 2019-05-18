import os
import numpy as np
from python_speech_features import mfcc,logfbank,delta, fbank
import scipy.io.wavfile as wav
from tqdm import tqdm


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


def extract_mfcc_psf(wav_path, feature_path, use_sdc=True, use_scale=True):
    """
    取 MFCC + 一阶 + 二阶 （13+13+13）或者 MFCC-SDC（13+49）
    """
    if not os.path.exists(feature_path):
        
        tmp = 'tmp'
        
        if not os.path.exists(tmp):
            os.makedirs(tmp)
        
        tmp_filename = 'tmp/__tmp__.wav'

        os.system('sox {} -c 1 -r 16000 -b 16 {}'.format(wav_path, tmp_filename))

        (rate, sig) = wav.read(tmp_filename)

        mfcc_feat = mfcc(sig, samplerate=16000, winlen=0.025, winstep=0.01, numcep=13,
                         nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97,
                         ceplifter=22, appendEnergy=True)
        if use_sdc:
            sdc = compute_sdc(mfcc_feat)

            mfcc_feat = normalize_frames(mfcc_feat, Scale=use_scale)
            sdc = normalize_frames(sdc, Scale=use_scale)
            mfccs_sdc_62 = np.hstack([mfcc_feat[:,:7], sdc])

#             use_sdc: (378, 13) (378, 62) (nframes,fetures)
            print('use_sdc:', mfcc_feat.shape, mfccs_sdc_62.shape)
            np.save(feature_path, mfccs_sdc_62)

        else:
            d_mfcc = delta(mfcc_feat, N=1)
            dd_mfcc = delta(d_mfcc, N=1)

            mfcc_feat = normalize_frames(mfcc_feat, Scale=use_scale)
            d_mfcc = normalize_frames(d_mfcc, Scale=use_scale)
            dd_mfcc = normalize_frames(dd_mfcc, Scale=use_scale)

            mfccs_39 = np.hstack([mfcc_feat, d_mfcc, dd_mfcc])

            # mfccs_39: (378, 39)  (nframes,fetures)
            print('mfccs_39:', mfccs_39.shape)
            np.save(feature_path, mfccs_39)
        os.remove(tmp_filename)


if __name__ == '__main__':

    data_dir = "./features/MFCC_SDC/"

#     data_dir = "./feature/MFCC_39/"

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # make train feature
    data_path = "data_list.txt"
    with open(data_path, 'r') as f:
        datafile = f.read().splitlines()
    for line in tqdm(datafile):

        filename = line.split("/")[-1].split('.wav')[0]

        label = line.split("/")[-3]

        filename = label +'_' + filename + '.npy'
        extract_mfcc_psf(os.path.join(line), os.path.join(data_dir, filename),
                         use_sdc=True, use_scale=True)
        break
    print('Extract Data MFCC Features Done!')