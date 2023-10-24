import os
import PIL
from PIL import ImageOps
import torch
import random
import torchvision
import numpy as np
import cv2

original_image_path         = "D:\\data\\ip102\\image\\0\\IP000000302.jpg"
original_image              = PIL.Image.open(original_image_path).convert('RGB')
Laplacian_diag              = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
# cv2/.imwrite("D:\\data\\ip102\\image2\\real_image.jpg",np.array(original_image))

original_image              = np.array(ImageOps.grayscale(original_image))
cv2.imwrite("D:\\data\\ip102\\image2\\gray_image.jpg",original_image)

edge                        = cv2.filter2D(original_image,-1, Laplacian_diag)
cv2.imwrite("D:\\data\\ip102\\image2\\edge_image.jpg",edge)