#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 15:01:40 2022

@author: negin
"""
from nets import DeepPyram_Res50 as Net

Net1 = Net

Categories = ['fold0','fold1','fold2','fold3','fold4']
Learning_Rates_init = [0.005]
epochs = 40
batch_size = 4
size = (512, 512)

Dataset_Path = '../../../../Datasets/Pupil_Iris_Segmentation_Cat3K/five_fold_dataset/'
mask_folder = '/masks/'
Results_path = '../results/'
Visualization_path = 'visualization_PupilID/'
Checkpoint_path = 'checkpoints_PupilID/'
CSV_path = 'CSVs_PupilID/'
project_name = "PupilID_V2"

load = False
load_path = ''

net_name = 'DeepPyram_Res50__'
test_per_epoch = 4




