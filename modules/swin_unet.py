# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from .swin_transformers import SwinTransformerSys

logger = logging.getLogger(__name__)

class SwinUnet(nn.Module):
    def __init__(self, **kwargs):
        super(SwinUnet, self).__init__()

        self.swin_unet = SwinTransformerSys(**kwargs)

    def forward(self, x):
        logits = self.swin_unet(x)
        return logits
