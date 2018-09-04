'''
load thumos 14 spatial temporal attention features, labels and image names

Author: Lili Meng
Date:  August 29th, 2018

'''
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from PIL import Image
from torchvision import transforms
import numpy as np
import torch

def make_label_dict():

    class_index_list = "./feature_list/new_ucf101_24_class_index.txt"

    lines = [line.strip() for line in open(class_index_list).readlines()]

    class_dict = {}
    for line in lines:
        class_label = line.split(' ')[0]
        class_name = line.split(' ')[1]
        class_dict[class_name] = class_label

    return class_dict

class UCF101AnnoDataset(Dataset):
    def __init__(self, category_dict, data_dir, csv_file, transform=None):
        self.data_dir = data_dir
        self.dataset = pd.read_csv(csv_file)
        self.category_dict = category_dict
      
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        feature_file = os.path.join(self.data_dir,  self.dataset['Feature'][idx])
        
        label_file = os.path.join(self.data_dir, self.dataset['Feature'][idx].replace("features", "label"))
        name_file = os.path.join(self.data_dir,  self.dataset['Feature'][idx].replace("features", "name"))
        
        feature_per_video = np.load(feature_file)
        
        name_per_video = np.load(name_file)

        video_name = name_per_video[0].split('/')[-2].split('_')[1]
        
        label_per_video = np.expand_dims(int(self.category_dict[video_name]), axis=0)
    
        sample = {'feature': feature_per_video, 'label': label_per_video}
        
        return sample, list(name_per_video)


def get_loader(class_dict, data_dir, csv_file, batch_size, mode='train', dataset='ucf101_24_anno'):
    """Build and return data loader."""

    
    shuffle = True if mode == 'train' else False

    if dataset == 'ucf101_24_anno':
        dataset = UCF101AnnoDataset(class_dict, data_dir, csv_file)
    else:
        raise Exception("no such dataset provided, currently only suppport ucf101_24_anno")
    
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

    return data_loader

if __name__ == '__main__':
    
    batch_size = 30

    class_dict = make_label_dict()
    print("class_dict")
    print(class_dict)
    train_data_dir = '/media/dataDisk/Video/spatial_temporal_att_ucf101_24/train_cnn_extract_features/spa_features/train/'
    train_csv_file = './feature_list/feature_train_list.csv'
    train_data_loader = get_loader(class_dict, train_data_dir, train_csv_file, batch_size=batch_size, mode='train',
                             dataset='ucf101_24_anno')
   
    for i, (train_sample, train_batch_name) in enumerate(train_data_loader):
        train_batch_feature = train_sample['feature']
        train_batch_label = train_sample['label']
        train_batch_name = np.swapaxes(np.asarray(train_batch_name), 0, 1)

        print("train_batch_feature.shape: ", train_batch_feature.shape)
        print("train_batch_name.shape: ", train_batch_name.shape)
        print("train_batch_label.shape: ", train_batch_label.shape)
        
        print("i: ", i)
        break
       
    test_data_dir = '/media/dataDisk/Video/spatial_temporal_att_ucf101_24/train_cnn_extract_features/spa_features/test/'
    test_csv_file = './feature_list/feature_test_list.csv'
    test_data_loader = get_loader(class_dict, test_data_dir, test_csv_file, batch_size=batch_size, mode='test',
                             dataset='ucf101_24_anno')
    
    all_test_names = []
    for i, (test_sample, test_batch_name) in enumerate(test_data_loader):
        test_batch_feature = test_sample['feature']
        test_batch_label = test_sample['label']
        test_batch_name = np.swapaxes(np.asarray(test_batch_name),0,1)
        print("test_batch_feature.shape: ", test_batch_feature.shape)
        print("test_batch_name.shape: ", test_batch_name.shape)
        print("test_batch_label.shape: ", test_batch_label.shape)
        all_test_names.append(test_batch_name)
        print("i: ", i)
    all_test_names = np.asarray(all_test_names)

    print("all_test_names.shape: ", all_test_names.shape)
    print("all_test_names[0].shape: ", all_test_names[0].shape)
        
      