import torch
from torch import nn
from torchvision.models.vgg import vgg16


class DescriminatorLoss(nn.Module):
    def __init__(self):
        super(DescriminatorLoss, self).__init__()
    def forward(self, fake_SR_D_result, real_Hr_D_result):
        tmp = fake_SR_D_result + real_Hr_D_result
        loss = 1 - tmp
        return  loss

class netELoss(nn.Module):
    def __init__(self):
        super(netELoss, self).__init__()
        self.mse = nn.MSELoss()

        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        
    def forward(self,New_HR_image,HR_image):

        mse_loss = self.mse(New_HR_image,HR_image)
        perception_loss = self.mse(self.loss_network(New_HR_image), self.loss_network(HR_image))
        
        return mse_loss + 0.002 * perception_loss 
        
class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
        self.L1Loss = nn.L1Loss()
    def forward(self, fake_SR_D_result, predicted_fake_Sr, Hr_image, middle_feature,x7):
        # Image Loss
        image_loss = self.mse_loss(predicted_fake_Sr, Hr_image)
        # Adversarial Loss
        adversarial_loss = torch.mean(1 - fake_SR_D_result)
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(predicted_fake_Sr), self.loss_network(Hr_image))
        # TV Loss
        tv_loss = self.tv_loss(predicted_fake_Sr)

        middle_feature = self.upsample(middle_feature)
        # L1 Loss
        feature_loss = self.L1Loss(middle_feature,x7)

        return image_loss + 0.001 * adversarial_loss  + 0.002 * perception_loss + 2e-8 * tv_loss + feature_loss * 0.001


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


if __name__ == "__main__":
    g_loss = GeneratorLoss()
    print(g_loss)
