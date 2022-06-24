#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 15:01:40 2022

@author: negin
"""
from nets import DeepPyram_ResNet50 as Net

Net1 = Net

# Add the paths to other categories if you have more than one folds for training and evaluation
Categories = ['fold0']

Learning_Rates_init = [0.005]
epochs = 40
batch_size = 4
size = (512, 512)

Pyramid_Loss=False

Dataset_Path = 'Datasets/DeepPyram/instruments/'
mask_folder = '/masks/'
Results_path = 'results/'
Visualization_path = 'visualization_DeepPyram/instruments'
Checkpoint_path = 'checkpoints_DeepPyram/instruments'
CSV_path = 'CSVs_DeepPyram/instruments'
project_name = "DeepPyram"

load = False
load_path = ''

net_name = 'DeepPyram_Res50__'
test_per_epoch = 4




