#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Spencer Perkins

TMRPCEN frontend

"""

import torch
import torch.nn as nn
import numpy as np


class TMRPCEN(nn.Module):
    """Trainable Multi-rate PCEN
    Args:
        n_bands : int, number of input frequency bands
        alpha : float, AGC exponent
        delta : float, DRC bias
        r : float, DRC exponent
        eps : float, small value to prevent division by 0
    """
    def __init__(self, n_bands: int=128, alpha: float=0.8,
                 delta: float=10., r: float=0.25, eps: float=10e-6):
        super(TMRPCEN, self).__init__()

        # Logs for trainable parameters
        s1 = np.log(np.abs(np.random.normal(0.03, 0.1)))
        s2 = np.log(np.abs(np.random.normal(0.03, 0.1)))
        s3 = np.log(np.abs(np.random.normal(0.03, 0.1)))
        s4 = np.log(np.abs(np.random.normal(0.03, 0.1)))
        s5 = np.log(np.abs(np.random.normal(0.03, 0.1)))
        s6 = np.log(np.abs(np.random.normal(0.03, 0.1)))
        s7 = np.log(np.abs(np.random.normal(0.03, 0.1)))
        s8 = np.log(np.abs(np.random.normal(0.03, 0.1)))
        s9 = np.log(np.abs(np.random.normal(0.03, 0.1)))
        s10 = np.log(np.abs(np.random.normal(0.03, 0.1)))
        
        alpha = np.log(alpha)
        delta = np.log(delta)
        r = np.log(r)

        self.s1 = nn.Parameter(torch.full((n_bands,), float(s1)))
        self.s2 = nn.Parameter(torch.full((n_bands,), float(s2)))
        self.s3 = nn.Parameter(torch.full((n_bands,), float(s3)))
        self.s4 = nn.Parameter(torch.full((n_bands,), float(s4)))
        self.s5 = nn.Parameter(torch.full((n_bands,), float(s5)))
        self.s6 = nn.Parameter(torch.full((n_bands,), float(s6)))
        self.s7 = nn.Parameter(torch.full((n_bands,), float(s7)))
        self.s8 = nn.Parameter(torch.full((n_bands,), float(s8)))
        self.s9 = nn.Parameter(torch.full((n_bands,), float(s9)))
        self.s10 = nn.Parameter(torch.full((n_bands,), float(s10)))
        
        self.alpha = nn.Parameter(torch.full((n_bands,), float(alpha)))
        self.delta = nn.Parameter(torch.full((n_bands,), float(delta)))
        self.r = nn.Parameter(torch.full((n_bands,), float(r)))
        self.eps = torch.as_tensor(eps)

    def forward(self, x):
        # x shape : [batch size, channels, frequency bands, time samples]
        # Exponentials of trainable parameters
        s_1 = self.s1.exp()
        s_2 = self.s2.exp()
        s_3 = self.s3.exp()
        s_4 = self.s4.exp()
        s_5 = self.s5.exp()
        s_6 = self.s6.exp()
        s_7 = self.s7.exp()
        s_8 = self.s8.exp()
        s_9 = self.s9.exp()
        s_10 = self.s10.exp()
        
        alpha = self.alpha.exp()
        delta = self.delta.exp()
        r = self.r.exp()

        # Broadcast over channel dimension
        alpha = alpha[:, np.newaxis]
        delta = delta[:, np.newaxis]
        r = r[:, np.newaxis]

        # Smoothing coefficient values
        s_vals = [s_1, s_2, s_3, s_4, s_5, s_6, s_7, s_8, s_9, s_10]

        # Storage for each PCEN computation
        layered_pcen = []

        # TMRPCEN

        # Compute the Smoothed filterbank
        for s in s_vals:
            # Smoother
            smoother = [x[..., 0]] # Initialize with first frame
            for frame in range(1, x.shape[-1]):
                smoother.append((1-s)*smoother[-1]+s*x[..., frame])
            smoother = torch.stack(smoother, -1)

            # Reformulation for (E / (eps + smooth)**alpha +delta)**r - delta**r
            # Vincent Lostenlan

            # Adaptive Gain Control
            smooth = torch.exp(-alpha*(torch.log(self.eps)+
                                     torch.log1p(smoother/self.eps)))
            # Dynamic Range Compression
            pcen_ = (x * smooth + delta)**r - delta**r

            # Store PCEN from current s value
            layered_pcen.append(pcen_)

        # Stack all computed PCEN 'layers'
        pcen_out = torch.stack(layered_pcen)
        pcen_out = torch.squeeze(pcen_out)

        # Reshape [Channels, batch_size, frequency bands, time samples]
        pcen_out = torch.permute(pcen_out, (1,0,2,3))

        return pcen_out
