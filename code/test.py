import math
import torch

from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from dataset import testdataset

import pytorch_ssim

from model_unet import Unet
from model import Generator, UpsampleBLock


if __name__ == '__main__':

##################### define hyper parameters

    UPSCALE_FACTOR = 4
    NUM_BATCH = 1
    SAVE_PATH = 'E:\\new_model(Unet + SRGAN)\\Edge_AutoEncoder+SRGAN_checkpoint\\Laplacian_diag\\Loss_0.01\\'
    EPOCH_LIST_TO_LOAD = list(range(200,210,10))#list(range(10,210,10))

##################### define data loader

    test_set = testdataset('E:\\IP102_public\\hourglass_2stack\\data\\coco\\test2017', upsampling_factor=UPSCALE_FACTOR)
    test_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=NUM_BATCH)

    num_of_batch = len(test_loader)

##################### define model

    unet_list = []
    netG_list = []

    for epoch_to_load in EPOCH_LIST_TO_LOAD:

        netG = Generator(UPSCALE_FACTOR)
        netG.load_state_dict(torch.load(SAVE_PATH + 'checkpoint/epoch_' + str(epoch_to_load) + '_ip102data_netG.pth'))
        netG.eval()
        for param in netG.parameters(): param.requires_grad = False
        if torch.cuda.is_available(): netG.cuda()
        
        netG_list.append(netG)

##################### start test

    niter = 0
    total_psnr = [ 0 for _ in range(len(EPOCH_LIST_TO_LOAD)) ]
    total_ssim = [ 0 for _ in range(len(EPOCH_LIST_TO_LOAD)) ]

    test_bar = tqdm(test_loader, total=num_of_batch, ascii=True)

    for lr, hr, path, w, h in test_bar:

        niter += 1

        lr = Variable(lr).cuda() if torch.cuda.is_available() else Variable(lr)
        hr = Variable(hr).cuda() if torch.cuda.is_available() else Variable(hr)

        for i in range(len(EPOCH_LIST_TO_LOAD)):
        
            x7,sr      = netG_list[i](lr)

            pixmax  = hr.max()**2
            mse     = ((sr - hr) ** 2).mean()
            psnr    = 10 * math.log10( pixmax / mse )
            total_psnr[i]+=psnr

            ssim    = pytorch_ssim.ssim(sr, hr)
            total_ssim[i]+=ssim

    total_psnr = [ i/niter for i in total_psnr ]
    total_ssim = [ i/niter for i in total_ssim ]

    print('\t\tepoch\t\tpsnr\t\tssim')
    for i in range(len(EPOCH_LIST_TO_LOAD)):
        print('\t\t%d\t\t%.3f\t\t%.4f' % (EPOCH_LIST_TO_LOAD[i], total_psnr[i], total_ssim[i]))