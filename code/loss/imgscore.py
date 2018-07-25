import utility
from model import common
from loss import discriminator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import random
import os
import scipy.io as sio
from PIL import Image
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class ImgScore(Dataset):
    def __init__(self, score_file, root_dir, transform=False):
        self.score_file = score_file
        self.root_dir = root_dir
        self.dir_img = os.path.join(root_dir, "img/")
        self.transform = transform

        #load val structure
        mat_contents = sio.loadmat(os.path.join(self.root_dir, self.score_file))
        oct_struct = mat_contents['scores_per']
        self.val = oct_struct[0,0]
        
    def __len__(self):
        length = 15989
        return length
    
    def img_normalizer(path):
        normalized_img = io.imread(path)
        return normalized_img
    
    def __getitem__(self,idx):
        strindex = str(idx).zfill(5)
        img_index = "id" + strindex[0:4] + "p" + strindex[4:5]
        filename = img_index[2:] + ".png"
        
        
        img = io.imread(os.path.join(self.dir_img, filename))

        fper = self.val[img_index]

        sample = {'image': img, 'f_per': int(fper)}

        if self.transform:
            sample = self.transform(sample)
        
        return sample

class ToTensor(object):
    def __call__(self, sample):
        image = sample['image']
        image = image.transpose((2,0,1))
        return {'image':torch.from_numpy(image), 'f_per':sample['f_per']}
        
