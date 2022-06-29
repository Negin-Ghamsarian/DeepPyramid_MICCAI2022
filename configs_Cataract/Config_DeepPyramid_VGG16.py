#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 15:01:40 2022

@author: negin
"""
from nets import DeepPyramid_VGG16 as Net

Net1 = Net

# Add the paths to other categories if you have more than one folds for training and evaluation
Categories = ['fold0']

Learning_Rates_init = [0.005]
epochs = 40
batch_size = 4
size = (512, 512)

Pyramid_Loss=False

Dataset_Path = 'Datasets/DeepPyramid/instruments/'
mask_folder = '/masks/'
Results_path = 'results/'
Visualization_path = 'visualization_DeepPyramid/instruments'
Checkpoint_path = 'checkpoints_DeepPyramid/instruments'
CSV_path = 'CSVs_DeepPyramid/instruments'
project_name = "DeepPyramid"

load = False
load_path = ''

net_name = 'DeepPyramid_VGG16__'
test_per_epoch = 4



