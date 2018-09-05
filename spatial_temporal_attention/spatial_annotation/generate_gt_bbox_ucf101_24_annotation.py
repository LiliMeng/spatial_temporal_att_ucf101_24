'''
Generate ground truth bounding boxes for ucf101 24 spatial annotation

The matlab version ground truth data is from 
https://github.com/gurkirt/corrected-UCF101-Annots

Author: Lili Meng
Date: Sep 4th, 2018
'''
import scipy.io as sio
import os

import subprocess
import numpy as np
import cv2


# def generate_IOU(gt_bbox, pred_bbox, img, img_index, save_folder):
#     # determine the (x, y)-coordinates of the intersection rectangle
#     xA = max(gt_bbox[0], pred_bbox[0])
#     yA = max(gt_bbox[1], pred_bbox[1])
#     xB = min(gt_bbox[2], pred_bbox[2])
#     yB = min(gt_bbox[3], pred_bbox[3])
    
#     # compute the area of intersection rectangle
#     interArea = (xB - xA + 1) * (yB - yA + 1)
    
#     # compute the area of both the prediction and ground-truth
#     # rectangles
#     boxAArea = (gt_bbox[2] - gt_bbox[0] + 1) * (gt_bbox[3] - gt_bbox[1] + 1)
#     boxBArea = (pred_bbox[2] - pred_bbox[0] + 1) * (pred_bbox[3] - pred_bbox[1] + 1)
    
#     # compute the intersection over union by taking the intersection
#     # area and dividing it by the sum of prediction + ground-truth
#     # areas - the interesection area
#     IOU = interArea / float(boxAArea + boxBArea - interArea)

#     img1 = img.copy()
    
#     cv2.rectangle(img1,(int(gt_bbox[0]),int(gt_bbox[1])),(int(gt_bbox[2]),int(gt_bbox[3])),(255,0,0),2)
#     cv2.rectangle(img1,(int(pred_bbox[0]),int(pred_bbox[1])),(int(pred_bbox[2]),int(pred_bbox[3])),(0,0,255),2)
#     cv2.imwrite(save_folder+"/bbox1_{}.png".format(img_index), img1)
    
#     # return the intersection over union value
# 	return IOU

def pick_24_video_classes():

	mat_dir = "finalAnnots"

	data = sio.loadmat(mat_dir)['annot']

	all_videos = data['name']

	annot_video_list = []
	for i in range(all_videos.shape[1]):
		print(i, all_videos[0][i][0])
		video_name = all_videos[0][i][0]
		annot_video_list.append(video_name)

	return annot_video_list

def pick_24_anno_classes():

	mat_dir = "finalAnnots"

	data = sio.loadmat(mat_dir)['annot']

	all_video_names = data['name']
	all_bboxes = data['tubes']
	all_video_img_nums = data['num_imgs']

	gt_bbox_dir = '/media/dataDisk/Video/spatial_temporal_att_ucf101_24/spatial_temporal_attention/mask_visualization/Contrast_0.0001_TV_reg1e-05_mask_LRPatience5_Adam0.0001_decay0.0001_dropout_0.2_Temporal_ConvLSTM_hidden512_regFactor_1_Sep_03_17_02/test_gt_bbox/'
	org_img_dir = "/media/dataDisk/THUMOS14/UCF101_jpegs_256/"

	for i in range(all_video_names.shape[1]):
		video_name = all_video_names[0][i][0].split('/')[1]

		print("video_name: ", video_name)
		end_frame = all_bboxes[0][i][0][0][0]

		start_frame = all_bboxes[0][i][0][0][1]
		bboxes_per_video = all_bboxes[0][i][0][0][3]
		per_video_bbox_dict = {}
		sub_gt_bbox_dir = os.path.join(gt_bbox_dir, video_name)
		if not os.path.exists(sub_gt_bbox_dir):
			os.makedirs(sub_gt_bbox_dir)
		for j in range(bboxes_per_video.shape[0]):

			per_frame_name = 'frame%06d.jpg'%(j+int(start_frame))
			frame_name = os.path.join(video_name, per_frame_name)
			per_frame_gt_bbox = bboxes_per_video[j]

			per_video_bbox_dict[per_frame_name] = per_frame_gt_bbox
			print("per_frame_name: ", per_frame_name)
			print("per_frame_bbox: ", per_frame_gt_bbox)

			org_img_name = os.path.join(org_img_dir, frame_name)
			print("org_img_name: ", org_img_name)
			org_img_copy = cv2.imread(org_img_name).copy()
			
			per_frame_gt_bbox[2] += per_frame_gt_bbox[0]
			per_frame_gt_bbox[3] += per_frame_gt_bbox[1]
			cv2.rectangle(org_img_copy,(int(per_frame_gt_bbox[0]),int(per_frame_gt_bbox[1])),(int(per_frame_gt_bbox[2]),int(per_frame_gt_bbox[3])),(255,0,0),2)
			cv2.imwrite(os.path.join(sub_gt_bbox_dir, per_frame_name), org_img_copy)
		



pick_24_anno_classes()