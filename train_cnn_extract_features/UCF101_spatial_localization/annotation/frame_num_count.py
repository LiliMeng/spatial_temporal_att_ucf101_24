'''
Count number of frames per video folder

Author: Lili Meng
Date: Sep 2nd, 2018
'''

import os
import numpy as np


def make_label_dict():

	class_index_list = "./classInd.txt"

	lines = [line.strip() for line in open(class_index_list).readlines()]

	class_dict = {}
	for line in lines:
		class_label = line.split(' ')[0]
		class_name = line.split(' ')[1]
		class_dict[class_name] = class_label

	return class_dict

def make_video_with_num_list():

	class_dict = make_label_dict()

	video_dir_list = "./ucf101_24_anno_train_list.txt"

	lines = [line.strip() for line in open(video_dir_list).readlines()]

	new_video_list = "new_ucf101_24_train.txt"
	new_file_with_img_num = open(new_video_list, "a")

	for line in lines:
		class_name = line.split('/')[0]
		videoname = line.split('/')[1]


		label = int(class_dict[class_name])-1
				
		videoname_dir = os.path.join("/media/dataDisk/THUMOS14/UCF101_jpegs_256", videoname)
		num_files = str(len(os.listdir(videoname_dir)))

		new_file_with_img_num.write(videoname+ " "+str(label)+" "+num_files+"\n")
		
		print("number_files: ", num_files)


make_video_with_num_list()