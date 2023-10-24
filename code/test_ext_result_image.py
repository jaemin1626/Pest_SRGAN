import os
import math
from pickle import FALSE
import torch
 
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from dataset import testdataset
import torchvision
import pytorch_ssim

from model_unet import Unet
from model import Generator, UpsampleBLock
from PIL import Image
import numpy as np
import cv2

def tensor_numpy(image,h,w):
        #image = torch.nn.functional.interpolate(image,  size = (h, w), mode = 'nearest').expand(-1, 3,-1,-1)
        grid = make_grid(image)
        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im = Image.fromarray(ndarr)
        #im.save("E:\\new_model(Unet + SRGAN)\\Edge_AutoEncoder+SRGAN_checkpoint\\Laplacian_diag\\Loss_0.01\\a.png")

        return ndarr

if __name__ == '__main__':

##################### define hyper parameters

    UPSCALE_FACTOR = 4
    NUM_BATCH = 1

    SAVE_PATH = 'E:\\new_model(Unet + SRGAN)\\Edge_AutoEncoder+SRGAN_checkpoint\\Laplacian_diag\\Loss_0.01\\'
    EPOCH_TO_LOAD = 200
    TEST_RESULT_PATH = SAVE_PATH + 'data_result/'
    TEST_RESULT_PATH_ = TEST_RESULT_PATH + 'with_' + str(EPOCH_TO_LOAD) + '_epoch_checkpoint_result/'
    TEST_RESULT_LR_PATH = TEST_RESULT_PATH_ + 'LR/'
    TEST_RESULT_SR_PATH = TEST_RESULT_PATH_ + 'SR/'
    TEST_RESULT_HR_PATH = TEST_RESULT_PATH_ + 'HR/'
    TEST_RESULT_SEG_PATH = TEST_RESULT_PATH_ + 'seg/'
    TEST_RESULT_lR_PATH = TEST_RESULT_PATH_ + "low_resolution/"
    SAVE_LR = True
    SAVE_HR = True
    SAVE_SEG= False
    SAVE_lR = True
##################### making save dir

    if not os.path.isdir(TEST_RESULT_PATH):
        os.makedirs(TEST_RESULT_PATH)
        print('TEST_RESULT_PATH dir created')
    if not os.path.isdir(TEST_RESULT_PATH_):
        os.makedirs(TEST_RESULT_PATH_)
        print(str(EPOCH_TO_LOAD) + '_result_path dir created')
    if not os.path.isdir(TEST_RESULT_LR_PATH) and SAVE_LR:
        os.makedirs(TEST_RESULT_LR_PATH)

    if not os.path.isdir(TEST_RESULT_LR_PATH) and SAVE_lR:
        os.makedirs(TEST_RESULT_lR_PATH)
        print('LR dir created')
    if not os.path.isdir(TEST_RESULT_SR_PATH):
        os.makedirs(TEST_RESULT_SR_PATH)
        print('SR dir created')
    if not os.path.isdir(TEST_RESULT_HR_PATH) and SAVE_HR:
        os.makedirs(TEST_RESULT_HR_PATH)
        print('HR dir created')
    if not os.path.isdir(TEST_RESULT_SEG_PATH) and SAVE_SEG:
        os.makedirs(TEST_RESULT_SEG_PATH)
        print('seg dir created')

##################### define data loader

    test_set = testdataset('E:\\IP102_public\\hourglass_2stack\\data\\coco\\test2017', upsampling_factor=UPSCALE_FACTOR)
    test_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=NUM_BATCH)


    num_of_batch = len(test_loader)
    totensor = torchvision.transforms.ToTensor()
##################### define model

    netG = Generator(UPSCALE_FACTOR)

    # upsample_block_num = int(math.log(UPSCALE_FACTOR, 2))
    # block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
    # block8.append(torch.nn.Conv2d(64, 3, kernel_size=9, padding=4))
    # block8 = torch.nn.Sequential(*block8)
    # netG.add_module(name='block8', module=block8)

    # unet.load_state_dict(torch.load(SAVE_PATH + 'checkpoint/epoch_' + str(EPOCH_TO_LOAD) + '_ip102data_unet.pth'))
    netG.load_state_dict(torch.load(SAVE_PATH + 'checkpoint/epoch_' + str(EPOCH_TO_LOAD) + '_ip102data_netG.pth'))
    netG.eval()
    for param in netG.parameters(): param.requires_grad = False
    if torch.cuda.is_available(): netG.cuda()

    ##################### start test

    niter = 0

    test_bar = tqdm(test_loader, total=num_of_batch, ascii=True)

    for lr, hr, path, w, h in test_bar:

        niter += 1

        path = path[0].split('/')
        name = path[-1].split('.')[0]

        lr = Variable(lr).cuda() if torch.cuda.is_available() else Variable(lr)
        hr = Variable(hr).cuda() if torch.cuda.is_available() else Variable(hr)

        a,sr      = netG(lr)

        pixmax  = hr.max()**2
        #mse     = ((sr - hr) ** 2).mean()
        #psnr    = 10 * math.log10( pixmax / mse )
        #ssim    = pytorch_ssim.ssim(sr, hr)

        Lr_image_       = torch.nn.functional.interpolate(lr,  size = (h, w), mode = 'nearest').expand(-1, 3,-1,-1)

        save_image(sr[0], TEST_RESULT_SR_PATH + '%s_.png'  % (name))
        if SAVE_LR :    save_image(Lr_image_[0], TEST_RESULT_LR_PATH + '%s.png'  % (name))
        if SAVE_HR :    save_image(hr[0], TEST_RESULT_HR_PATH + '%s.png' % (name))
        if SAVE_lR :    save_image(lr[0], TEST_RESULT_lR_PATH + '%s.png'  % (name))

        # data_lr = Image.open(TEST_RESULT_lR_PATH + '%s.png' % (name)).convert('RGB')
        
        # h = data_lr.height
        # w = data_lr.width

        # image_t = totensor(data_lr).unsqueeze(0)
        # image_t = Variable(image_t).cuda() if torch.cuda.is_available() else Variable(image_t)

        # a, sr      = netG(image_t)
        # image_t = tensor_numpy(sr,h,w)
        # image_t = np.array(image_t)
        # image_t = cv2.cvtColor(image_t, cv2.COLOR_RGB2BGR)
        
        # cv2.imshow('test', image_t)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()                 
        # print(image_t)