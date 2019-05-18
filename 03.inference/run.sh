#!/bin/bash

# ================================================================
find "/home/zhudong/PycharmProjects/FM_Radio_Detecting_329/01.getFeatures/features/LogFbank"  -name  '*.npy'  >data_list_lfb.txt

perl get_list.pl data_list_lfb.txt lanKey.txt label_data_list_lfb.txt

python3 inference.py