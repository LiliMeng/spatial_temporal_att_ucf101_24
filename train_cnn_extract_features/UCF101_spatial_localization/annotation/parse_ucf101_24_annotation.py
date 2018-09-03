'''
Parse UCF101 spatial localization annotation
Author: Lili Meng
Date: Sep 1st, 2018
'''
import scipy.io as sio
import os

import subprocess
import numpy as np


def pick_24_video_classes():

	mat_dir = "finalAnnots"

	data = sio.loadmat(mat_dir)['annot']

	all_videos = data['name']

	#annota_video_list = open("./ucf101_24_anno_list.txt", "a")

	annot_video_list = []
	for i in range(all_videos.shape[1]):
		print(i, all_videos[0][i][0])
		video_name = all_videos[0][i][0]
		annot_video_list.append(video_name)
		#annota_video_list.write(all_videos[0][i][0]+"\n") 

	return annot_video_list

def pick_24_anno_classes():

	mat_dir = "finalAnnots"

	data = sio.loadmat(mat_dir)['annot']

	all_videos = data['name']

	#annota_video_list = open("./ucf101_24_anno_list.txt", "a")

	annot_video_list = []
	for i in range(all_videos.shape[1]):
		print(i, all_videos[0][i][0])
		video_name = all_videos[0][i][0].split('/')[0]
		annot_video_list.append(video_name)
		#annota_video_list.write(all_videos[0][i][0]+"\n") 

	annot_video_list = list(set(annot_video_list))

	return annot_video_list


def ucf101_org_list(video_list):

	#train_list = "/media/dataDisk/Video/two-stream-action-recognition/UCF_list/trainlist01.txt"

	lines = [line.strip() for line in open(video_list).readlines()]

	train_videoname_list = []
	for line in lines:
		video_name = line.split('.avi')[0]
		train_videoname_list.append(video_name)

	return train_videoname_list


def make_anno_test_video_list():

	ucf101_24_annot_video_list = pick_24_video_classes()

	ucf101_test_list = ucf101_org_list("/media/dataDisk/Video/spatial_temporal_att_ucf101_24/train_cnn_extract_features/UCF_list/old_testlist01.txt")

	print("len(ucf101_24_annot_video_list) ", len(ucf101_24_annot_video_list))
	print("ucf101_24_annot_video_list[0] ", ucf101_24_annot_video_list[0])

	print("len(ucf101_test_list): ", len(ucf101_test_list))
	print("ucf101_test_list[0] ", ucf101_test_list[0])


	elems_in_anno_test_lists = set(ucf101_24_annot_video_list) & set(ucf101_test_list)

	print("len(elems_in_anno_test_lists) ", len(elems_in_anno_test_lists))

	annot_test_video_list = open("./ucf101_24_anno_test_list.txt", "a")

	for i in range(len(list(elems_in_anno_test_lists))):

		test_video_name = sorted(list(elems_in_anno_test_lists))[i]
		print("test_video_name: ", test_video_name)

		annot_test_video_list.write(test_video_name+"\n") 


def make_anno_train_video_list():

	ucf101_train_list = ucf101_org_list("/media/dataDisk/Video/spatial_temporal_att_ucf101_24/train_cnn_extract_features/UCF_list/trainlist01.txt")

	print("len(ucf101_train_list) ", len(ucf101_train_list))
	print("ucf101_train_list[0] ", ucf101_train_list[0])

	anno_classes=pick_24_anno_classes()

	print("anno_classes")
	print(anno_classes)
	print("len(anno_classes): ", len(anno_classes))

	annot_train_video_list = open("./ucf101_24_anno_train_list.txt", "a")

	for i in range(len(ucf101_train_list)):
		if ucf101_train_list[i].split('/')[0] in anno_classes:
			train_video_name = ucf101_train_list[i]
			annot_train_video_list.write(train_video_name+"\n")

make_anno_train_video_list()