
import torch
import torch.nn as nn
import numpy as np
import torchvision
import os

os.environ['TORCH_HOME'] = "/home/xyb/Lab/TorchModels"

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.emodel = torchvision.models.efficientnet_b7(pretrained=True)
        # Change first conv layer to accept single-channel (grayscale) input
        #self.resnet.conv1.weight = torch.nn.Parameter(self.resnet.conv1.weight.sum(dim=1).unsqueeze(1)) 
        
    def forward(self, x):
        skip_connections = []
        for i in range(9):
            x = list(self.emodel.children())[0][i](x)
            skip_connections.append(x)
        encoder_outputs = skip_connections.pop(-1)
        #skip_connections = skip_connections[::-1]
        
        return encoder_outputs, skip_connections

class AttentionUNet(nn.Module):
    """
    The Attention-UNet implementation based on PaddlePaddle.
    As mentioned in the original paper, author proposes a novel attention gate (AG)
    that automatically learns to focus on target structures of varying shapes and sizes.
    Models trained with AGs implicitly learn to suppress irrelevant regions in an input image while
    highlighting salient features useful for a specific task.
    The original article refers to
    Oktay, O, et, al. "Attention u-net: Learning where to look for the pancreas."
    (https://arxiv.org/pdf/1804.03999.pdf).
    Args:
        num_classes (int): The unique number of target classes.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self, num_classes, pretrained=None):
        super().__init__()
        n_channels = 3
        self.encoder = FeatureExtractor()
        filters = np.array([64, 32, 48, 80, 160, 224, 384, 640, 2560])
        ##                   0   1   2   3    4     5   6    7     8



        self.up9 = UpConv(ch_in=filters[8], ch_out=filters[7], upsample=False)
        self.att9 = AttentionBlock(
            F_g=filters[7], F_l=filters[7], F_out=filters[6])
        self.up_conv9 = ConvBlock(ch_in=2*filters[7], ch_out=filters[7])

        self.up8 = UpConv(ch_in=filters[7], ch_out=filters[6], upsample=False)
        self.att8 = AttentionBlock(
            F_g=filters[6], F_l=filters[6], F_out=filters[5])
        #self.up_conv8 = ConvBlock(ch_in=filters[7], ch_out=filters[6])
        self.up_conv8 = ConvBlock(ch_in=2*filters[6], ch_out=filters[6])

        self.up7 = UpConv(ch_in=filters[6], ch_out=filters[5])
        self.att7 = AttentionBlock(
            F_g=filters[5], F_l=filters[5], F_out=filters[4])
        self.up_conv7 = ConvBlock(ch_in=2*filters[5], ch_out=filters[5])

        self.up6 = UpConv(ch_in=filters[5], ch_out=filters[4], upsample=False)
        self.att6 = AttentionBlock(
            F_g=filters[4], F_l=filters[4], F_out=filters[3])
        self.up_conv6 = ConvBlock(ch_in=2*filters[4], ch_out=filters[4])



        self.up5 = UpConv(ch_in=filters[4], ch_out=filters[3])
        self.att5 = AttentionBlock(
            F_g=filters[3], F_l=filters[3], F_out=filters[2])
        self.up_conv5 = ConvBlock(ch_in=2*filters[3], ch_out=filters[3])

        self.up4 = UpConv(ch_in=filters[3], ch_out=filters[2])
        self.att4 = AttentionBlock(
            F_g=filters[2], F_l=filters[2], F_out=filters[1])
        self.up_conv4 = ConvBlock(ch_in=2*filters[2], ch_out=filters[2])

        self.up3 = UpConv(ch_in=filters[2], ch_out=filters[1])
        self.att3 = AttentionBlock(
            F_g=filters[1], F_l=filters[1], F_out=filters[0])
        self.up_conv3 = ConvBlock(ch_in=2*filters[1], ch_out=filters[1])

        self.up2 = UpConv(ch_in=filters[1], ch_out=filters[0], upsample=False)
        self.att2 = AttentionBlock(
            F_g=filters[0], F_l=filters[0], F_out=filters[0] // 2)
        self.up_conv2 = ConvBlock(ch_in=2*filters[0], ch_out=filters[0])

        self.up1 = UpConv(ch_in=filters[0], ch_out=num_classes)

        self.conv_1x1 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, stride=1, padding=0)
        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        x9, (x1, x2, x3, x4, x5, x6, x7, x8) = self.encoder(x)



        d9 = self.up9(x9)
        x8 = self.att9(g=d9, x=x8)
        d9 = torch.concat([x8, d9], axis=1)
        d9 = self.up_conv9(d9)

        d8 = self.up8(x8)
        x7 = self.att8(g=d8, x=x7)
        d8 = torch.concat([x7, d8], axis=1)
        d8 = self.up_conv8(d8)

        d7 = self.up7(d8)
        x6 = self.att7(g=d7, x=x6)
        d7 = torch.concat([x6, d7], axis=1)
        d7 = self.up_conv7(d7)

        d6 = self.up6(d7)
        x5 = self.att6(g=d6, x=x5)
        d6 = torch.concat([x5, d6], axis=1)
        d6 = self.up_conv6(d6)




        d5 = self.up5(d6)
        x4 = self.att5(g=d5, x=x4)
        d5 = torch.concat([x4, d5], axis=1)
        d5 = self.up_conv5(d5)

        d4 = self.up4(d5)
        x3 = self.att4(g=d4, x=x3)
        d4 = torch.concat((x3, d4), axis=1)
        d4 = self.up_conv4(d4)

        d3 = self.up3(d4)
        x2 = self.att3(g=d3, x=x2)
        d3 = torch.concat((x2, d3), axis=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        x1 = self.att2(g=d2, x=x1)
        d2 = torch.concat((x1, d2), axis=1)
        d2 = self.up_conv2(d2)

        logit = self.up1(d2) #self.conv_1x1(d2)
        #logit_list = [logit]
        return logit #logit_list

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_out):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(
                F_g, F_out, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_out))

        self.W_x = nn.Sequential(
            nn.Conv2d(
                F_l, F_out, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_out))

        self.psi = nn.Sequential(
            nn.Conv2d(
                F_out, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid())

        self.relu = nn.ReLU()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        res = x * psi
        return res


class UpConv(nn.Module):
    def __init__(self, ch_in, ch_out, upsample=True, kernel_size=3, stride=1, padding=1):
        super().__init__()
        if upsample:
            self.up = nn.Sequential(
                nn.Upsample(
                    scale_factor=2, mode="bilinear"),
                nn.Conv2d(
                    ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(ch_out),
                nn.ReLU())
        else:
            self.up = nn.Sequential(
                nn.Conv2d(
                    ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(ch_out),
                nn.ReLU())

    def forward(self, x):
        return self.up(x)

class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),
            nn.Conv2d(
                ch_out, ch_out, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(ch_out),
            nn.ReLU())

    def forward(self, x):
        return self.conv(x)
