'''
Generate heatmap for UCF101 24 classes spatial annotation

Author: Lili Meng
Date: Sep 3rd, 2018
'''
import os
import numpy as np
import cv2

def generate_heatmap(mask_dir, mode, use_heatmap):

    mode_mask_dir = os.path.join(mask_dir, mode)

    saved_vis_dir = os.path.join("./mask_visualization",mode_mask_dir)

    if not os.path.exists(saved_vis_dir):
        os.makedirs(saved_vis_dir)

    video_name= np.load("./saved_weights/"+mask_dir+"/"+mode.split('_')[0]+"_name.npy")
    video_weights = np.load("./saved_weights/"+mask_dir+"/"+mode.split('_')[0]+"_att_weights.npy")

    video_name = np.concatenate(video_name, axis=0)

    #print("video_name.shape: ", video_name.shape)

    video_frame_length = video_name.shape[1]
    print("video_frame_length: ", video_frame_length)

    for video_index in range(video_name.shape[0]):
        single_video_name = video_name[video_index]
        single_video_weights = video_weights[video_index].reshape(video_frame_length,7,7)


        for img_indx in range(video_frame_length):
            img_path = single_video_name[img_indx]
            print('img_path: ', img_path)
           
            img = cv2.imread(img_path)
            #print("img_path: ", img_path)
            height, width, _ = img.shape
            print("height ", height)
            print("width ", width)
            img = cv2.resize(img, (width, height))

            cam = single_video_weights[img_indx]
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
            cam_img = np.uint8(255 * cam_img)
            cam_img = cv2.resize(cam_img, (width, height))

            if use_heatmap == True:
                heatmap = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
                result = heatmap * 0.3 + img* 0.5
            else:
                result = cam_img


            result_dir = os.path.join(saved_vis_dir, img_path.split('/')[-2])
            result_name = img_path.split('/')[-1]
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            #print("result_name: ", result_name)
            cv2.imwrite(result_dir+'/'+result_name, result)


mask_dir ="Contrast_0.0001_TV_reg1e-05_mask_LRPatience5_Adam0.0001_decay0.0001_dropout_0.2_Temporal_ConvLSTM_hidden512_regFactor_1_Sep_03_17_02"
mode = "test_gray"

if mode == "test_heatmap":
    use_heatmap = True
elif mode == "test_gray":
    use_heatmap = False
else:
    raise Exception("only test_heatmap and test_gray mode are provided now")
generate_heatmap(mask_dir,mode, use_heatmap)
