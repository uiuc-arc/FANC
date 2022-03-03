import torch
import torchvision
import torchvision.transforms as transforms
from model_defs import model_cnn_3layer, model_mlp_any, model_cnn_4layer, model_cnn_2layer, model_cnn_2layer_op, model_cnn_5layer, model_cnn_4layer_conv13, model_cnn_4layer_conv11
import numpy as np
import torch.nn as nn
import os

import sys 
sys.path.insert(0, './diffai')

from models import *  

DEVICE = 'cpu'


checkpoint_name = 'diffai/defended-networks/basic_nets/MNIST/width_0.3/ConvSmall__LinMix_a_IFGSM_w_Lin_00.410020__k_3__b_InSamp_Lin_01502__w_Lin_00.415050___bw_Lin_00.515050___checkpoint_401_with_0.969.pynet'

# ex_input = inputs[0].reshape(1, 1, 28, 28)
skip_layer = 0
    
model = torch.load(checkpoint_name, map_location=torch.device(DEVICE))
