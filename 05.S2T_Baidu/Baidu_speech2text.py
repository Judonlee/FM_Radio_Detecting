# -*- coding: utf-8 -*-

"""
@Author : zhudong
@Email  : ynzhudong@163.com
@Time   : 2019/4/2 上午8:55
@File   : Baidu_speech2text.py
desc:
"""

import os
from tqdm import tqdm
from aip import AipSpeech


APP_ID = ""
API_KEY = ""
SECRET_KEY = ""
client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)


# 读取文件
def get_file_content(filePath):
    tmp = 'tmp'

    if not os.path.exists(tmp):
        os.makedirs(tmp)

    tmp_filename = 'tmp/__tmp__.wav'

    os.system('sox {} -c 1 -r 16000 -b 16 {}'.format(filePath, tmp_filename))
    with open(tmp_filename, 'rb') as fp:
        signal = fp.read()

    os.remove(tmp_filename)

    return signal


def main(wavlist):
    # make data feature
    with open(wavlist, 'r') as f:
        datafile = f.read().splitlines()

    speech2text_result = {}
    for line in tqdm(datafile):
        # 识别本地文件
        sig_result = client.asr(get_file_content(line), 'wav', 16000, {'dev_pid': 1536})
        print(sig_result)
        speech2text_result[line] = sig_result
    return speech2text_result


if __name__ == '__main__':

    wavpath = './speech_signal'
    datalist = 'data_list.txt'
    os.system("find {} -name '*.wav' | sort > {}".format(wavpath, datalist))
    text_dic= main(datalist)

    print('\n************************ 音频转录后的结果 **********************')

    for idx, (key, value) in enumerate(text_dic.items()):

        if value["err_no"] == 0:
            print('\n音频文件：{} , 对应翻译为： {}'.format(key, value['result']))

        else:
            # value["err_no"] == 3301:
            print('\n文件 {} 由于 {} 无法翻译！'.format(key, value['err_msg']))

    print('\n*************************************************************')
