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
    print(filePath)

    os.system('sox {} -c 1 -r 16000 -b 16 {}'.format(filePath, tmp_filename))
    with open(tmp_filename, 'rb') as fp:
        signal = fp.read()

    os.remove(tmp_filename)

    return signal


# 识别本地文件
result = client.asr(get_file_content('speech_signal/91.8MHz.wav'), 'wav', 16000, {'dev_pid': 1536})

print(result)

