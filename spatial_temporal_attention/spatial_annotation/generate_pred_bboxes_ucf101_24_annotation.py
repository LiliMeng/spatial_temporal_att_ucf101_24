'''
Generate bounding boxes for UCF101 24 classes annotation

Author: Lili Meng

Date: Sep 3rd, 2018
'''

import numpy as np
import cv2
import os
import shutil
import csv
import pickle

def generate_boundingbox(gray_img, threshold):
    """Generate a bounding box for the heatmap"""
    
    gray = gray_img.copy()
    ret,th1 = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    _, contours, hierarchy = cv2.findContours(th1, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    size = 0
    x, y, w, h = 0, 0, 0, 0
    for contour in contours:
        x_, y_, w_, h_ = cv2.boundingRect( contour )
        if w_ * h_ > size:
            x, y, w, h = x_, y_, w_, h_
            size = w * h

    return [x, y, x+w, y+h]

    
# def generate_boundingbox(gray_img, threshold):

# 	result_gray_img_show = gray_img.copy()

# 	result_gray_img_show = result_gray_img_show - result_gray_img_show.min()
# 	result_gray_img_show = result_gray_img_show/result_gray_img_show.max()
# 	result_gray_img_show *= 255

# 	ret,th1 = cv2.threshold(result_gray_img_show, threshold, 255, cv2.THRESH_BINARY)
# 	height, width = th1.shape[:2]
# 	x1, y1, x2, y2 = width*2, height*2, 0, 0
# 	for y in range(height):
# 		for x in range(width):
# 			if th1[y,x] != 0:
# 				x1, y1 = min(x1, x), min(y1, y)
# 				x2, y2 = max(x2, x), max(y2, y)
# 	if x1 == width*2:
# 		print("empty boundingbox!")
# 		return [0, 0, 0, 0]
# 	else:
# 		return [x1, y1, x2, y2]


def draw_boundingbox_on_heatmap_img(gray_img_video_dir, heatmap_img_video_dir, boundingbox_dir, bbox_threshold):
	
	if not os.path.exists(boundingbox_dir):
		os.makedirs(boundingbox_dir)

	video_pred_bbox_dict = {}
	for sub_dir in sorted(os.listdir(gray_img_video_dir)):
		sub_gray_img_video_dir = os.path.join(gray_img_video_dir, sub_dir)
		
		sub_heatmap_img_video_dir = os.path.join(heatmap_img_video_dir, sub_dir)
		sub_bounding_box_dir = os.path.join(boundingbox_dir, sub_dir)
		assert(len(os.listdir(sub_gray_img_video_dir))==20)
		assert(len(os.listdir(sub_heatmap_img_video_dir))==20)
		if not os.path.exists(sub_bounding_box_dir):
			os.makedirs(sub_bounding_box_dir)
		for img_name in sorted(os.listdir(sub_gray_img_video_dir)):
			assert(len(os.listdir(sub_gray_img_video_dir))==20)
			if '.jpg' in img_name:

				gray_img = cv2.imread(os.path.join(sub_gray_img_video_dir,  img_name), 0)
				pred_box = generate_boundingbox(gray_img, bbox_threshold)

				frame_name = os.path.join(sub_dir, img_name)
				video_pred_bbox_dict[frame_name] = pred_box
				org_img = cv2.imread(os.path.join(sub_heatmap_img_video_dir, img_name))
				org_img_copy = org_img.copy()
				cv2.rectangle(org_img_copy,(int(pred_box[0]),int(pred_box[1])),(int(pred_box[2]),int(pred_box[3])),(255,0,0),2)
				cv2.imwrite(os.path.join(sub_bounding_box_dir, img_name), org_img_copy)

	print("len(video_pred_bbox_dict): ", len(video_pred_bbox_dict))
	pickle_out = open("video_pred_bbox_dict.pickle","wb")
	pickle.dump(video_pred_bbox_dict, pickle_out)
	pickle_out.close()
	

gray_img_video_dir = '/media/dataDisk/Video/spatial_temporal_att_ucf101_24/spatial_temporal_attention/mask_visualization/Contrast_0.0001_TV_reg1e-05_mask_LRPatience5_Adam0.0001_decay0.0001_dropout_0.2_Temporal_ConvLSTM_hidden512_regFactor_1_Sep_03_17_02/test_gray'
heatmap_img_video_dir = '/media/dataDisk/Video/spatial_temporal_att_ucf101_24/spatial_temporal_attention/mask_visualization/Contrast_0.0001_TV_reg1e-05_mask_LRPatience5_Adam0.0001_decay0.0001_dropout_0.2_Temporal_ConvLSTM_hidden512_regFactor_1_Sep_03_17_02/test_heatmap'
boundingbox_dir = '/media/dataDisk/Video/spatial_temporal_att_ucf101_24/spatial_temporal_attention/mask_visualization/Contrast_0.0001_TV_reg1e-05_mask_LRPatience5_Adam0.0001_decay0.0001_dropout_0.2_Temporal_ConvLSTM_hidden512_regFactor_1_Sep_03_17_02/test_bbox/'
bbox_threshold = 200

draw_boundingbox_on_heatmap_img(gray_img_video_dir, heatmap_img_video_dir, boundingbox_dir, bbox_threshold)