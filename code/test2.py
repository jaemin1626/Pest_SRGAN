from BugSrgan import BugSrgan
import cv2
from torchvision.utils import save_image
from PIL import Image

original_image_path = "E:\\new_model(Unet + SRGAN)\\Edge_AutoEncoder+SRGAN_checkpoint\\Laplacian_diag\\Loss_0.01\\data_result\\with_200_epoch_checkpoint_result\\low_resolution\\IP000000064.png"
original_image = Image.open(original_image_path).convert('RGB')

srgan_model = BugSrgan()
res = srgan_model.make_sr(original_image)

cv2.imshow('test', res)
cv2.waitKey(0)
cv2.destroyAllWindows()