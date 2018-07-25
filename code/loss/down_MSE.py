from model import common

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from torchvision import transforms
#from PIL import Image

def bicubic_tensor_down(tensor):
    #tensor is size 16,3,192,192
    batch_size,n_channel,h,w = tensor.size()
    downed_tensor = torch.FloatTensor(batch_size,n_channel,h/4,w/4)
    for img_index, img in enumerate(tensor):
        size = (h/4,w/4)
        downed_img = transforms.Resize(size, interpolation=Image.BICUBIC)
        downed_tensor(img_index) = downed_img
    
    return downed_tensor

class down_MSE(nn.Module):
    def __init__(self):
        super(down_MSE, self).__init__()


    def forward(self, sr, hr):
        down_hr = 1
        down_sr = 1
        loss = F.mse_loss(hr,sr)

        return loss

"""           
transforms.Resize(size=[16, 3, 192/4, 192/4],interpolation=Image.BICUBIC)

LRTrans = transforms.Compose([
                  transforms.Scale(192 // 4, Image.BICUBIC),
                  transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                  ])     """
