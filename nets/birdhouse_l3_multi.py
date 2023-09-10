# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 11:24:41 2023

@author: Spencer perkins

Implementation of CNN model, a slightly modified version of L3 audio subnetwork
SUPPORTING MULTIPLE CHANNEL INPUT
"""

import torch
import torch.nn as nn

class birdHouseL3(nn.Module):
    def __init__(self):
        super(birdHouseL3, self).__init__()

        """Four Convloutional block implementation of L3 with some
            modifications"""

        # Block one : n in-chans for MRPCEN, 1 for logmel/pcen
        self.conv1 = nn.Sequential(
                        nn.Conv2d(10, 32, kernel_size=3, padding='same'),
                        nn.BatchNorm2d(32),
                        nn.ReLU()
                        )

        self.conv2 = nn.Sequential(
                        nn.Conv2d(32, 32, kernel_size=3, padding='same'),
                        nn.BatchNorm2d(32),
                        nn.ReLU()
                        )
        self.pool_b1 = nn.MaxPool2d(2,2)

        # Block two
        self.conv3 = nn.Sequential(
                        nn.Conv2d(32, 64, kernel_size=3, padding='same'),
                        nn.BatchNorm2d(64),
                        nn.ReLU()
                        )
        self.conv4 = nn.Sequential(
                        nn.Conv2d(64, 64, kernel_size=3, padding='same'),
                        nn.BatchNorm2d(64),
                        nn.ReLU()
                        )
        self.pool_b2 = nn.MaxPool2d(2,2)

        # Block three
        self.conv5 = nn.Sequential(
                        nn.Conv2d(64, 128, kernel_size=3, padding='same'),
                        nn.BatchNorm2d(128),
                        nn.ReLU()
                        )
        self.conv6 = nn.Sequential(
                        nn.Conv2d(128, 128, kernel_size=3, padding='same'),
                        nn.BatchNorm2d(128),
                        nn.ReLU()
                        )
        self.pool_b3 = nn.MaxPool2d(2,2)

        # Block 4
        self.conv7 = nn.Sequential(
                        nn.Conv2d(128, 256, kernel_size=3, padding='same'),
                        nn.BatchNorm2d(256),
                        nn.ReLU()
                        )
        self.conv8 = nn.Sequential(
                        nn.Conv2d(256, 256, kernel_size=3, padding='same'),
                        nn.BatchNorm2d(256),
                        nn.ReLU()
                        )
        self.pool_b4 = nn.MaxPool2d(2,2)

        # Linear layer + out layer
        self.fc = nn.Sequential(
                        nn.Linear(108544, 512),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(512, 1024),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(1024, 1),
                        )                   


    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool_b1(x)
        # Block 2
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool_b2(x)
        # Block 3
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool_b3(x)
        # Block 4
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.pool_b4(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        x= torch.sigmoid(x)
        x= torch.reshape(x, (-1,))

        return x
