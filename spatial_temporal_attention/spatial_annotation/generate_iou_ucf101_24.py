'''
Generate IOU for ucf101 24 classes annotation
Author: Lili Meng
Date: Sep 4th, 2018
'''
import numpy as np
import pickle
import cv2
import os

def generate_IOU(gt_bbox, pred_bbox, img, img_name, save_folder):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(gt_bbox[0], pred_bbox[0])
    yA = max(gt_bbox[1], pred_bbox[1])
    xB = min(gt_bbox[2], pred_bbox[2])
    yB = min(gt_bbox[3], pred_bbox[3])
    
    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    if interArea <0:
    	interArea = 0
    
    # compute the area of both the prediction and ground-truth
    # rectangles
    gt_bboxArea = (gt_bbox[2] - gt_bbox[0] + 1) * (gt_bbox[3] - gt_bbox[1] + 1)
    pred_bboxArea = (pred_bbox[2] - pred_bbox[0] + 1) * (pred_bbox[3] - pred_bbox[1] + 1)
    
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    IOU = interArea / float(gt_bboxArea + pred_bboxArea - interArea)


    img1 = img.copy()
    
    cv2.rectangle(img1,(int(gt_bbox[0]),int(gt_bbox[1])),(int(gt_bbox[2]),int(gt_bbox[3])),(255,0,0),2)
    cv2.rectangle(img1,(int(pred_bbox[0]),int(pred_bbox[1])),(int(pred_bbox[2]),int(pred_bbox[3])),(0,0,255),2)
    final_save_folder = os.path.join(save_folder, img_name.split('/')[0])

    if not os.path.exists(final_save_folder):
    	os.makedirs(final_save_folder)
    cv2.imwrite(os.path.join(final_save_folder, img_name.split('/')[1]), img1)
    
    # return the intersection over union value
    return IOU


gt_bbox_pickle_in = open("video_gt_bbox_dict.pickle","rb")
video_gt_bbox_dict = pickle.load(gt_bbox_pickle_in)

pred_bbox_pickle_in = open("video_pred_bbox_dict.pickle","rb")
video_pred_bbox_dict = pickle.load(pred_bbox_pickle_in)

print("len(video_gt_bbox_dict): ", len(video_gt_bbox_dict))
print("len(video_pred_bbox_dict): ", len(video_pred_bbox_dict))

gt_bbox_set = set(video_gt_bbox_dict.keys())
pred_bbox_set = set(video_pred_bbox_dict.keys())

same_video_frame_gt_pred = gt_bbox_set.intersection(pred_bbox_set)

print("len(same_video_frame_gt_pred)")
print(len(same_video_frame_gt_pred))

heatmap_dir = '/media/dataDisk/Video/spatial_temporal_att_ucf101_24/spatial_temporal_attention/mask_visualization/Contrast_0.0001_TV_reg1e-05_mask_LRPatience5_Adam0.0001_decay0.0001_dropout_0.2_Temporal_ConvLSTM_hidden512_regFactor_1_Sep_03_17_02/test_heatmap'

iou_threshold = 0.3
correct_iou_num =0
for frame_name in same_video_frame_gt_pred:
	gt_bbox = video_gt_bbox_dict[frame_name]
	pred_bbox = video_pred_bbox_dict[frame_name]

	img = cv2.imread(os.path.join(heatmap_dir,frame_name))
	per_frame_iou=generate_IOU(gt_bbox, pred_bbox, img, frame_name, "iou")
	print("per_frame_iou: ", per_frame_iou)
	if per_frame_iou >= iou_threshold:
		correct_iou_num += 1

correct_iou_percen = correct_iou_num/len(same_video_frame_gt_pred)

print("correct_iou_percen: ", correct_iou_percen)

