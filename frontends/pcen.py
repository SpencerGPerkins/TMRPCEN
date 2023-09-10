#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 13:48:35 2023

@author: Spencer Perkins

Per-channel Energy Normalization with trainable alpha, delta, r frontend

"""

import torch
import torch.nn as nn
import numpy as np


class PCEN(nn.Module):
    """Apply Trainable Per-channel energy Normalization (train Alpha, delta, r)
    Args:
        n_bands : int, number of input frequency bands
        t_val : float, value for t -> s (smoothing coefficient)
        alpha : float, gain control
        delta : float, bias
        r : float,
        eps : float, small value to prevent division by 0
    """
    def __init__(self, n_bands: int=128, t_val : float= 2**8, alpha: float=0.8,
                 delta: float=10., r: float=0.25, eps: float=10e-6, print_p: bool=False):
        super(PCEN, self).__init__()

        alpha = np.log(alpha)
        delta = np.log(delta)
        r = np.log(r)

        self.t_val = torch.as_tensor(t_val)
        self.alpha = nn.Parameter(torch.full((n_bands,), float(alpha)))
        self.delta = nn.Parameter(torch.full((n_bands,), float(delta)))
        self.r = nn.Parameter(torch.full((n_bands,), float(r)))
        self.eps = torch.as_tensor(eps)
        self.print_p = print_p

    def forward(self, x):
        # x shape : [batch size, channels, frequency bands, time samples]
        # Calculate smoothing coefficient s
        s = (torch.sqrt(1 + 4* self.t_val**2) - 1) / (2 * self.t_val**2)
        alpha = self.alpha.exp()
        delta = self.delta.exp()
        r = self.r.exp()
        
        # Broadcast over channel dimension
        alpha = alpha[:, np.newaxis]
        delta = delta[:, np.newaxis]
        r = r[:, np.newaxis]

        # Smoother
        smoother = [x[..., 0]] # Initialize with first frame
        for frame in range(1, x.shape[-1]):
            smoother.append((1-s)*smoother[-1]+s*x[..., frame])
        smoother = torch.stack(smoother, -1)

        # Reformulation for (E / (eps + smooth)**alpha +delta)**r - delta**r
        # Vincent Lostenlan
        smooth = torch.exp(-alpha*(torch.log(self.eps)+
                                     torch.log1p(smoother/self.eps)))
        pcen_ = (x * smooth + delta)**r - delta**r
        pcen_out = pcen_.permute((0,1,3,2))

        return pcen_out
