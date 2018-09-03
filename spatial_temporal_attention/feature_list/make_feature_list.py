'''
Make feature list for the csv file needed by the feature_loader.py

Author: Lili Meng
Date: August 29th, 2018

'''

import numpy as np
import os


feature_dir = "/media/dataDisk/Video/spatial_temporal_att_ucf101_24/train_cnn_extract_features/spa_features/train/"
txt_file = open("feature_train_list.txt", mode='a')

txt_file.write("Feature"+"\n")


for feature_filename in sorted(os.listdir(feature_dir)):
	
	if 'features' in feature_filename:
		print("feature_filename: ", feature_filename)

		txt_file.write(feature_filename+"\n")