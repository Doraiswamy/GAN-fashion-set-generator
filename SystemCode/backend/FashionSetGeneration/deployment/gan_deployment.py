# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 17:31:11 2020

@author: Lakshmi Subramanian
"""

from __future__ import print_function
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt


def generator_test():
    # helper function to un-normalize and display an image

    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image

    # #### defining Generator block and loading the weights
    class G(nn.Module):

        def __init__(self):
            super(G, self).__init__()
            self.main = nn.Sequential(
                nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
                nn.Tanh()
            )

        def forward(self, input):
            output = self.main(input)
            return output

    # ###initialising new class
    netG1 = G()
    # ### loading pre trained weights
    netG1.load_state_dict(torch.load(os.getcwd()+"/FashionSetGeneration/deployment/generator.pth"))
    noise = Variable(torch.randn(16, 100, 1, 1))
    fake = netG1(noise)

    fig = plt.figure(figsize=(25, 4))
    # display 16 images
    for idx in np.arange(len(fake.data)):
        ax = fig.add_subplot(2, 16 / 2, idx + 1, xticks=[], yticks=[])
        imshow(fake.data[idx])
        
    fig.savefig('FashionSetGeneration/deployment/GANgenerator.jpg')
    return {'image_path': os.getcwd()+'/FashionSetGeneration/deployment/GANgenerator.jpg'}


