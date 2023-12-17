import gc
import os
import time
import math
import torch
import torch.optim as optim
import torch.utils.data
import pandas as pd
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms as TF
from dataset import traindataset
from torch import nn
import pytorch_ssim
import matplotlib.pyplot as plt
from loss import netELoss
from Autoencoder import Dehaze
import cv2
import numpy as np
import matplotlib.pyplot as plt
if __name__ == '__main__':

    CROP_SIZE = 128
    NUM_EPOCHS = 200
    START_EPOCH = 0
    UPSCALE_FACTOR = 4
    NUM_BATCH = 10

    OUT_PATH = 'E:\\new_model(Unet + SRGAN)\\Unet + SRGAN_AutoEncoder\\'
    CHECKPOINT_PATH = OUT_PATH+'checkpoint/'

    ## data 경로
    train_set = traindataset('E:\\Image_Captioning_data\\default\\train2014', crop_size=CROP_SIZE, upsampling_factor=UPSCALE_FACTOR)
    train_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=NUM_BATCH, shuffle=True)

    netE= Dehaze()    
    if torch.cuda.is_available():
        netE = Dehaze()
        netE.cuda() 
        
    netE.train()    
    criterionE = netELoss()
    if torch.cuda.is_available(): criterionE.cuda()

    optimizer_netE = optim.Adam(netE.parameters())
    

    start_time = int(time.time())

    results= {'total_loss' : [], 'feature_loss':[], 'psnr':[] }

    for epoch in range(START_EPOCH + 1, NUM_EPOCHS + 1):
        
        batch_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'hr_score': 0, 'sr_score': 0, 'mse': 0, 'ssim': 0, 'ssims': 0, 'psnr': 0}
        niter = 0

        for LR_image, HR_image in train_loader:

            niter += 1

            LR_image            = Variable(LR_image).cuda() if torch.cuda.is_available() else Variable(LR_image)
            HR_image            = Variable(HR_image).cuda() if torch.cuda.is_available() else Variable(HR_image)

            New_HR_image, HR_image_feature            = netE(HR_image)

            loss = criterionE(New_HR_image,HR_image)
            
