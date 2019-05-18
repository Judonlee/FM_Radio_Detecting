#!/bin/bash

# 获取.wav文件路径
find "../dataset/" -name '*.wav'  >data_list.txt


# #======================利用自己写的函数提取音频特征=====================
# # 提取特征并把特征保存features文件夹下

# python3 makeSpectrogram.py

# python3 makeMFCC.py

python3 makeLogFbanks.py

# python3 makePLP.py


# #=======================利用HTK提取音频特征============================
# # 暂时还没调试通
# #python3 make_htk.py