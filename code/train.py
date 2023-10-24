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
from torchvision.utils import save_image
from torchvision import transforms as TF
from torch import nn
import pytorch_ssim
from dataset import traindataset
from model import Generator, Discriminator
from Autoencoder import Dehaze
from loss import GeneratorLoss, DescriminatorLoss
import pandas
import torchvision
##################### define hyper parameters

CROP_SIZE = 128
NUM_EPOCHS = 200
START_EPOCH = 0
UPSCALE_FACTOR = 4
NUM_BATCH = 10

BASE_PATH = 'E:\\new_model(Unet + SRGAN)\\Edge_AutoEncoder+SRGAN_checkpoint\\Laplacian_diag\\Loss_0.01\\'
LOSS_PATH = BASE_PATH+'loss/'
CHECKPOINT_PATH = BASE_PATH+'checkpoint/'
TRAIN_RESULT_PATH = BASE_PATH+'train_result/'

##################### making save dir

if not os.path.exists(LOSS_PATH):
    os.makedirs(LOSS_PATH)
    print('loss dir created')
if not os.path.exists(CHECKPOINT_PATH):
    os.makedirs(CHECKPOINT_PATH)
    print('checkpoint dir created')
if not os.path.exists(TRAIN_RESULT_PATH):
    os.makedirs(TRAIN_RESULT_PATH)
    print('train dir created')

##################### define data loader

train_set = traindataset('D:\\data\\ip102\\image', crop_size=CROP_SIZE, upsampling_factor=UPSCALE_FACTOR)
train_loader = torch.utils.data.DataLoader(dataset=train_set, num_workers=0, batch_size=NUM_BATCH, shuffle=True)

num_of_batch = len(train_loader)


print('train data len : ', len(train_set))

##################### define model

netG = Generator(UPSCALE_FACTOR)
netD = Discriminator()


netE = Dehaze()

netE.load_state_dict(torch.load("E:\\new_model(Unet + SRGAN)\\Edge_TeacherNetwork\\AutoEncoder_Laplacian_diagonal\\checkpoint\\epoch_190_ip102data_netE.pth"))

for param in netE.parameters():
    param.requires_grad = False

if(START_EPOCH != 0):

    netG .load_state_dict(torch.load(CHECKPOINT_PATH + 'epoch_' + str(START_EPOCH) + '_ip102data_netG.pth'))
    netD .load_state_dict(torch.load(CHECKPOINT_PATH + 'epoch_' + str(START_EPOCH) + '_ip102data_netD.pth'))

    
if torch.cuda.is_available():
    device = torch.device("cuda")
if torch.cuda.is_available(): 
    netG.cuda()
if torch.cuda.is_available(): 
    netD.cuda()
if torch.cuda.is_available(): netE.cuda()
netG.train()
netD.train()

##################### define loss function

criterionG = GeneratorLoss()
criterionD = DescriminatorLoss()
if torch.cuda.is_available(): criterionG.cuda()
if torch.cuda.is_available(): criterionD.cuda()

##################### define optimizer Adam

optimizer_netG = torch.optim.Adam(netG.parameters())
optimizer_netD = torch.optim.Adam(netD.parameters())

##################### start training

start_time = int(time.time())

results = {'d_loss': [], 'g_loss': [], 'hr_score': [], 'sr_score': [], 'psnr': [], 'ssim': []}

for epoch in range(START_EPOCH + 1, NUM_EPOCHS + 1):

    batch_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'hr_score': 0, 'sr_score': 0, 'mse': 0, 'ssim': 0, 'ssims': 0, 'psnr': 0}
    niter = 0

    for LR_image, HR_image,edge in train_loader:

        niter += 1

        LR_image            = Variable(LR_image).cuda() if torch.cuda.is_available() else Variable(LR_image)
        HR_image            = Variable(HR_image).cuda() if torch.cuda.is_available() else Variable(HR_image)
        edge                = Variable(edge).cuda() if torch.cuda.is_available() else Variable(edge)

        x7,SR_image         = netG(LR_image)
        SR_hr_score         = netD(SR_image).mean()
        HR_hr_score         = netD(HR_image).mean()


        optimizer_netD.zero_grad()
        d_loss = criterionD(SR_hr_score, HR_hr_score)
        d_loss.backward(retain_graph=True)
        optimizer_netD.step()


        optimizer_netG.zero_grad()

        x7,SR_image             = netG(LR_image)
        SR_hr_score             = netD(SR_image).mean()
        out,middle_feature      = netE(edge)

        g_loss = criterionG(SR_hr_score, SR_image, HR_image, middle_feature, x7)
        g_loss.backward()
        optimizer_netG.step()


        batch_size = len(LR_image)
        batch_results['batch_sizes'] += batch_size
        batch_results['g_loss'     ] += g_loss.item() * batch_size
        batch_results['d_loss'     ] += d_loss.item() * batch_size
        batch_results['sr_score'   ] += SR_hr_score.item() * batch_size
        batch_results['hr_score'   ] += HR_hr_score.item() * batch_size

        batch_mse                   = ((SR_image - HR_image) ** 2).data.mean()
        batch_results['mse']        += batch_mse * batch_size
        batch_ssim                  = pytorch_ssim.ssim(SR_image, HR_image).item()
        batch_results['ssims']      += batch_ssim * batch_size
        batch_results['psnr']       = 10 * math.log10((HR_image.max()**2) / (batch_results['mse'] / batch_results['batch_sizes']))
        batch_results['ssim']       = batch_results['ssims'] / batch_results['batch_sizes']

        middle_time     = int(time.time())
        tm              = time.gmtime(middle_time - start_time)
        hour            = tm.tm_hour
        min             = tm.tm_min
        sec             = tm.tm_sec

        print('\r',end='')
        print('\033[93m'+f'-time[{hour}:{min}:{sec}]---epoch[{epoch}/{NUM_EPOCHS}]---iter[{niter}/{num_of_batch}]---'+
                '\033[93m'+'psnr[%.4f]---ssim[%.4f]---g_loss[%.4f]---d_loss[%.4f]---sr_score[%.4f]---hr_score[%.4f]-' % (
                batch_results['psnr'],
                batch_results['ssim'],
                batch_results['g_loss']     / batch_results['batch_sizes'],
                batch_results['d_loss']     / batch_results['batch_sizes'],
                batch_results['sr_score']   / batch_results['batch_sizes'],
                batch_results['hr_score']   / batch_results['batch_sizes']) + '\033[37m',end='')

        if niter == 1 or (int(num_of_batch * 1/3) == niter) or (int(num_of_batch * 2/3) == niter):
            
            Lr_image_       = torch.nn.functional.interpolate(LR_image, size = (CROP_SIZE, CROP_SIZE), mode = 'nearest').expand(-1, 3,-1,-1)

            zero_space      = torch.zeros(3, CROP_SIZE, 10)
            zero_space      = zero_space.cuda() if torch.cuda.is_available() else zero_space

            image_for_save  = [ torch.cat( (
                                            Lr_image_[i],
                                            SR_image[i],
                                            HR_image[i]
                                            ), 2) for i in range(len(Lr_image_)) ]
            image_for_save = torch.cat(image_for_save, 1)

            torchvision.utils.save_image(image_for_save, TRAIN_RESULT_PATH + 'epoch_%d_%d.png' % (epoch, niter))

    results['psnr'      ].append(batch_results['psnr'])
    results['ssim'      ].append(batch_results['ssim'])
    results['d_loss'    ].append(batch_results['d_loss']   / batch_results['batch_sizes'])
    results['g_loss'    ].append(batch_results['g_loss']   / batch_results['batch_sizes'])
    results['hr_score'  ].append(batch_results['hr_score'] / batch_results['batch_sizes'])
    results['sr_score'  ].append(batch_results['sr_score'] / batch_results['batch_sizes'])

    if epoch % 5 == 0:

        torch.save(netG.state_dict(), CHECKPOINT_PATH + 'epoch_%d_ip102data_netG.pth' % (epoch))
        torch.save(netD.state_dict(), CHECKPOINT_PATH + 'epoch_%d_ip102data_netD.pth' % (epoch))

        data_frame = pandas.DataFrame(data={
                                        'PSNR'    : results['psnr'],
                                        'SSIM'    : results['ssim'],
                                        'Loss_G'  : results['g_loss'],
                                        'Loss_D'  : results['d_loss'],
                                        'Score_sr': results['sr_score'],
                                        'Score_hr': results['hr_score']
                                        }, index=range(START_EPOCH + 1, epoch + 1))
        data_frame.to_csv(LOSS_PATH + 'epoch_%d_train_results.csv' % (epoch), index_label='Epoch')
    gc.collect()