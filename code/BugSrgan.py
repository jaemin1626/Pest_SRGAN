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
import PIL
import cv2
import torchvision
import numpy as np
from PIL import Image
UPSCALE_FACTOR = 4
from torchvision.utils import make_grid
class BugSrgan():
    def __init__(self):
        self.totensor = torchvision.transforms.ToTensor()
        self.h = None
        self.w = None  
        self.netG = Generator(UPSCALE_FACTOR)
        self.netG.load_state_dict(torch.load("E:\\new_model(Unet + SRGAN)\\Edge_AutoEncoder+SRGAN_checkpoint\\Laplacian_diag\\Loss_0.01\\checkpoint\\epoch_200_ip102data_netG.pth"))
        if torch.cuda.is_available(): self.netG.cuda()
        self.netG.eval()

    def tensor_numpy(self,image):
        image = torch.nn.functional.interpolate(image,  size = (self.h, self.w), mode = 'nearest').expand(-1, 3,-1,-1)
        grid = make_grid(image)
        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im = Image.fromarray(ndarr)
        #im.save("E:\\new_model(Unet + SRGAN)\\Edge_AutoEncoder+SRGAN_checkpoint\\Laplacian_diag\\Loss_0.01\\a.png")

        return ndarr
    
    def make_sr(self,image):
        image_t = self.totensor(image).unsqueeze(0)

        self.h = image.height
        self.w = image.width

        image_t = Variable(image_t).cuda() if torch.cuda.is_available() else Variable(image_t)
        
        a, sr      = self.netG(image_t)
        #save_image(sr[0],"E:\\new_model(Unet + SRGAN)\\Edge_AutoEncoder+SRGAN_checkpoint\\Laplacian_diag\\Loss_0.01\\a.png")
        
        img = self.tensor_numpy(sr)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  
        
        # cv2.imshow('test', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return img