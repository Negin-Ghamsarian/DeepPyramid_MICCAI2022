#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 13:43:53 2022

@author: negin
"""

import random 
import argparse
import logging 
import os 
import sys 
import csv
import numpy as np 
from tqdm import tqdm 
import wandb
import importlib

import torch 
import torch.nn as nn 
from torch import optim 
from torch.utils.data import DataLoader

from torchvision.transforms.functional import resize

from utils.eval_dice_IoU_binary import eval_dice_IoU_binary
from utils.save_metrics import save_metrics
from utils.dataset_PyTorch import BasicDataset
from utils.losses_binary_ReduceMean import DiceBCELoss
from utils.import_helper import import_config
from utils.seed_initialization import seed_all, seed_worker

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)


def create_directory(new_dir):
    try:
        os.mkdir(new_dir)
        logging.info('Created checkpoint directory')
    except OSError:
        pass

def train_net(net,
              epochs=30,
              batch_size=1,
              lr=0.001,
              device='cuda',
              save_cp=True,
              Pyramid_Loss=True,
              size = (512,512)
              ):

    TESTS = []
    train_dataset = BasicDataset(dir_train_img, dir_train_mask, size = size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False)
    n_train = len(train_dataset)
    inference_step = np.floor(np.ceil(n_train/batch_size)/test_per_epoch)
    print(f'Inference Step:{inference_step}')

    test_dataset = BasicDataset(dir_test_img, dir_test_mask)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=False)#, drop_last=True)
    n_test = len(test_dataset)


    
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Test size:       {n_test}
        Checkpoints:     {save_cp}
        Device:          {device.type}
    ''')

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 2, gamma = 0.9)

    criterion = DiceBCELoss()
    test_counter = 1
    for epoch in range(epochs):
        net.train()
        

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:

                imgs = batch['image']
                true_masks = batch['mask']

                if Pyramid_Loss:
                    mask3 = resize(true_masks, size//2)
                    mask2 = resize(mask3, size//4)
                    mask1 = resize(mask2, size//8)
                
            
                
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                
                
                if Pyramid_Loss:

                    masks_pred, mask1_pred, mask2_pred, mask3_pred = net(imgs)
                    loss_main = criterion(masks_pred, true_masks)
                    loss_wandb = loss_main
            
                    loss = loss_main
                    epoch_loss += loss.item()

                    loss1 = criterion(mask1_pred, mask1)
                    loss2 = criterion(mask2_pred, mask2)
                    loss3 = criterion(mask3_pred, mask3)

                    loss_main = loss_main + 0.75*loss3 + 0.5*loss2  + 0.25*loss1 



                else:    

                    masks_pred = net(imgs)
                    loss_main = criterion(masks_pred, true_masks)
                    loss_wandb = loss_main
            
                    loss = loss_main
                    epoch_loss += loss.item()


                

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
            
                (loss_main).backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                
                ########################################################################
                ########################################################################
                if (global_step) % (inference_step) == 0: # Should be changed if the condition that the n_train%batch_size != 0
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        

                    val1, val2, val3, val4, val5, val6, val7, val8, inference_time = eval_dice_IoU_binary(net, test_loader, device, test_counter, save_test, save=False, Pyramid_Loss=Pyramid_Loss)
                    
                    print(f'Validation Dice:{val1}')
                    print(f'Validation IoU:{val3}')
                    TESTS.append([val1, val2, val3, val4, val5, val6, val7, val8, inference_time, epoch_loss])

                    test_counter = test_counter+1
                    #scheduler.step(val_score)
                    

                    if net.n_classes > 1:
                         raise Exception("Not implemented for multi-class segmentation")
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val1))
                        logging.info('Validation IoU: {}'.format(val3))
                        
                    wandb.log({'Train_Loss': loss_wandb,
                            'Test_Dice': val1,
                            'Test_IoU': val3})
                    
                    

        scheduler.step()
           
        if save_cp:
            if True:
                try:
                    os.mkdir(dir_checkpoint)
                    logging.info('Created checkpoint directory')
                except OSError:
                    pass
                torch.save(net.state_dict(),
                           dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
                logging.info(f'Checkpoint {epoch + 1} saved !')
            
    val1, val2, val3, val4, val5, val6, val7, val8, inference_time = eval_dice_IoU_binary(net, test_loader, device, test_counter, save_test, save=True, Pyramid_Loss=Pyramid_Loss)
    save_metrics(TESTS, csv_name)
     

   


if __name__ == '__main__':

    args = parser.parse_args()
    config_file = args.config
    my_conf = importlib.import_module(config_file)
    Categories,Learning_Rates_init, epochs, batch_size, size,\
             Dataset_Path_Train, Dataset_Path_Test,\
                  mask_folder, Results_path, Visualization_path,\
                 CSV_path, project_name, load, load_path, net_name,\
                      test_per_epoch, Checkpoint_path, Net1, Pyramid_Loss\
                     = import_config.execute(my_conf)

    print("inside main")
    print('Hello Ubelix')
    print(f'Cuda Availability: {torch.cuda.is_available()}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device : {device}')
    print(f'Cuda Device Number: {torch.cuda.current_device()}')
    print(f'Cuda Device Name: {torch.cuda.get_device_name(0)}')
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Using device {device}')
    
    
    try:
        for c in range(len(Categories)):      
            for LR in range(len(Learning_Rates_init)):

                print(f'Initializing the learning rate: {Learning_Rates_init[LR]}')

                wandb.init(project=project_name+'_'+net_name+str(Learning_Rates_init[LR])+'_'+str(batch_size), entity="negin_gh",
                name="CATARACT")
                wandb.config = {
                "learning_rate": Learning_Rates_init[LR],
                "epochs": epochs,
                "batch_size": batch_size,
                "net_name": net_name,
                "Dataset": "Fold"+str(c),

                }
                

                dir_train_img = Dataset_Path_Train +'imgs'  
                dir_train_mask = Dataset_Path_Train + mask_folder 
                
                dir_test_img = Dataset_Path_Test+'imgs'
                dir_test_mask = Dataset_Path_Test + mask_folder
                
                save_test = Results_path + Visualization_path +'CATARACT_'+net_name +str(Learning_Rates_init[LR])+'_'+str(Categories[c])+'/'
                
                dir_checkpoint = Results_path + Checkpoint_path +'CATARACT_'+ net_name +str(Learning_Rates_init[LR])+'_'+str(Categories[c])+'/'
                csv_name = Results_path + CSV_path +'CATARACT_'+net_name +str(Learning_Rates_init[LR])+'_'+str(Categories[c])+'.csv'
                
                create_directory(Results_path + Visualization_path)
                create_directory(Results_path + Checkpoint_path)
                create_directory(Results_path + CSV_path)


                net = Net1(n_classes=1, n_channels=3, bilinear=True, Pyramid_Loss=Pyramid_Loss)
                logging.info(f'Network:\n'
                             f'\t{net.n_channels} input channels\n'
                             f'\t{net.n_classes} output channels (classes)\n')

                if load:
                    net.load_state_dict(
                        torch.load(load_path, map_location=device)
                    )
                    logging.info(f'Model loaded from {str(load_path)}')

                    with open(CSV_path) as csv_file:
                        csv_reader = csv.reader(csv_file, delimiter=',')
                        line_count = 0
                        for row in csv_reader:
                            if line_count > 0:
                               wandb.log({'Train_Loss': row[9]},{'Test_Dice': row[0]}, {'Test_IoU': row[2]})
                            line_count += 1  

                net.to(device=device)
                
                train_net(net=net,
                          epochs=epochs,
                          batch_size=batch_size,
                          lr=Learning_Rates_init[LR],
                          device=device,
                          Pyramid_Loss=Pyramid_Loss,
                          size = size)
            
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)