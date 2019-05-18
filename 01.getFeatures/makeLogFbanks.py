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


if __name__ == '__main__':

    # data_dir = "./features/LogFbank_3/"

    data_dir = "./features/LogFbank/"

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # make data feature
    data_path = "data_list.txt"
    with open(data_path, 'r') as f:
        datafile = f.read().splitlines()
    i = 1
    for line in tqdm(datafile):

        filename = line.split("/")[-1].split('.wav')[0]

        label = line.split("/")[-3]

        filename = label +'_' + filename + '.npy'
#         print(filename)
#         print(os.path.join(data_dir, filename))

#         extract_logfbank_psf(os.path.join(line), os.path.join(data_dir, filename), filter_bank=40,
#                              use_delta=False, use_logscale=True, use_scale=True)
        
        try:
            extract_logfbank_psf(os.path.join(line), os.path.join(data_dir, filename), filter_bank=40,
                             use_delta=False, use_logscale=True, use_scale=True)
        except:
            i += 1
            print("except wav_path", line)

        # break
    # print(i)
    print('Extract Data LogFbank Features Done!')
