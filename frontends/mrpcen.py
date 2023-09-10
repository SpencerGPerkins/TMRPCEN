#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 13:48:35 2023

@author: Spencer Perkins

Multi-rate PCEN frontend

"""

import torch
import torch.nn as nn
import numpy as np


class MRPCEN(nn.Module):
    """Apply Multi-rate Per-channel energy Normalization (trainable alpha, delta, r)
    Args:
        n_bands : int, number of input frequency bands
        t_values : list, values for t -> s (smoothing coefficient)
        alpha : float, gain control
        delta : float, bias
        r : float,
        eps : float, small value to prevent division by 0
    """
    def __init__(self, n_bands: int=128, t_values : list=[], alpha: float=0.8,
                 delta: float=10., r: float=0.25, eps: float=10e-6):
        super(MRPCEN, self).__init__()

        alpha = np.log(alpha)
        delta = np.log(delta)
        r = np.log(r)

        self.t_values = t_values
        self.alpha = nn.Parameter(torch.full((n_bands,), float(alpha)))
        self.delta = nn.Parameter(torch.full((n_bands,), float(delta)))
        self.r = nn.Parameter(torch.full((n_bands,), float(r)))
        self.eps = torch.as_tensor(eps)

    def forward(self, x):
        # x shape : [batch size, channels, frequency bands, time samples]
        t_tens_vals =[]
        for i in range(len(self.t_values)):
            t_tens_vals.append(torch.as_tensor(self.t_values[i]))
        s_values = []
        for t in t_tens_vals:
            s = (torch.sqrt(1 + 4* t**2) - 1) / (2 * t**2)
            s_values.append(torch.as_tensor(s))
        alpha = self.alpha.exp()
        delta = self.delta.exp()
        r = self.r.exp()
        # Broadcast over channel dimension
        alpha = alpha[:, np.newaxis]
        delta = delta[:, np.newaxis]
        r = r[:, np.newaxis]

        layered_pcen = []

        # Multi-rate PCEN
        for s in s_values:
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
            layered_pcen.append(pcen_)
        pcen_out = torch.stack(layered_pcen)
        pcen_out = torch.squeeze(pcen_out)


        pcen_out = torch.permute(pcen_out, (1,0,2,3))

        return pcen_out
