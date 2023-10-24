from torch.utils.data.dataset import Dataset
import os
from PIL import Image
import torchvision.transforms as TF
import torch.nn.functional as f
import random
import cv2

class traindataset(Dataset):
    def __init__(self, data_path, crop_size, upsampling_factor) -> None:
        super(traindataset).__init__()

        self.Crop_size = crop_size
        self.Upsampling_Factor = upsampling_factor

        self.original_image_file_name_list = []

        for x in os.listdir(data_path):
            x_path = data_path + '/' + x
            self.original_image_file_name_list.append(x_path)

        self.totensor = TF.ToTensor()
        self.Hflip    = TF.RandomHorizontalFlip(1)
        self.Vflip    = TF.RandomVerticalFlip(1)

    def random_crop(self, input_original_image, crop_size):
        w = input_original_image.width
        h = input_original_image.height

        if w < crop_size :
            input_original_image    = input_original_image.resize( (crop_size, h), Image.BILINEAR)
            w = input_original_image.width
        if h < crop_size :
            input_original_image    = input_original_image.resize( (w, crop_size), Image.BILINEAR)
            h = input_original_image.height
        
        w_ = random.randint(0, w-crop_size)
        h_ = random.randint(0, h-crop_size)

        input_original_image_       = input_original_image.crop((w_, h_, w_ + crop_size, h_ + crop_size))
        
        return input_original_image_

    def __getitem__(self, index):
        original_image_path         = self.original_image_file_name_list[index]
        original_image              = Image.open(original_image_path).convert('RGB')

        original_image              = self.random_crop(original_image, self.Crop_size)
        original_image              = self.Vflip(original_image)
        original_image              = self.Hflip(original_image)

        Hr_image                    = self.totensor(original_image)
        Lr_image                    = original_image.resize((self.Crop_size//self.Upsampling_Factor, self.Crop_size//self.Upsampling_Factor), Image.BILINEAR)
        Lr_image                    = self.totensor(Lr_image)

        return Lr_image, Hr_image

    def __len__(self):
        return len(self.original_image_file_name_list)

class testdataset(Dataset):
    def __init__(self, data_path, upsampling_factor) -> None:
        super(traindataset).__init__()

        self.upsampling_factor = upsampling_factor

        self.original_image_file_name_list = []

        for x in os.listdir(data_path):
            x_path = data_path + '/' + x
            self.original_image_file_name_list.append(x_path)

        print('##################### Test set size : %d' % len(self.original_image_file_name_list))

        self.totensor = TF.ToTensor()

    def __getitem__(self, index):
        original_image_path = self.original_image_file_name_list[index]
        original_image = Image.open(original_image_path).convert('RGB')

        #original_image = cv2.imread(self.original_image_file_name_list[index])
        w = original_image.width
        h = original_image.height
        
        Hr_image                    = self.totensor(original_image)
        Lr_image                    = self.totensor(original_image.resize((w//self.upsampling_factor, h//self.upsampling_factor), Image.BILINEAR))
        
        return Lr_image, Hr_image, original_image_path, w, h

    def __len__(self):
        return len(self.original_image_file_name_list)