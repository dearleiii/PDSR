import utility
from model import common
from loss import discriminator
from loss import imgscore

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import random
import os
import scipy.io as sio
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class Adversarial(nn.Module):
    def __init__(self, args, gan_type):
        super(Adversarial, self).__init__()
        self.gan_type = gan_type
        self.gan_k = args.gan_k
        self.aprx_epochs = args.aprx_epochs
        self.aprx_training_dir = args.aprx_training_dir
        self.aprx_training_dir_HR = args.aprx_training_dir_HR
        self.batch_size = args.batch_size
        self.patch_size = args.patch_size
        self.discriminator = discriminator.Discriminator(args, gan_type)
        self.optimizer = utility.make_optimizer(args, self.discriminator)
        self.scheduler = utility.make_scheduler(args, self.optimizer)
        self.a_counter = 0

    def forward(self, fake, real):
        #print('Counter is at: ', str(self.a_counter))
        fake_detach = fake.detach()

        self.loss = 0
        for _ in range(self.gan_k):
            self.optimizer.zero_grad()   
            if self.a_counter<self.aprx_epochs:
                dataset = imgscore.ImgScore(score_file = 'combined_struct_continues.mat',
                                   root_dir ="/usr/xtmp/superresoluter/approximater_training_set/patches/",
                                   transform=imgscore.ToTensor()
                                    )
                train_loader = torch.utils.data.DataLoader(dataset, batch_size = self.batch_size,
                                                      shuffle = True, num_workers = 4)
                for i, data in enumerate(train_loader):
                    img_batch = data['image'].float()
                    img_batch = img_batch.cuda(async=True)
                    img_fper = data['f_per'].float()
                    img_fper = img_fper.cuda(async=True)
                    
                    img_fper = torch.unsqueeze(img_fper, 1)
                    
                    break
                #print('img size: ', img_batch.size(), img_batch.type(), 'f_per size:', \
                 #         img_fper.size(), img_fper.type(), 'dis output:', loss_g.size(), \
                  #        loss_g.type(), 'fake size:', fake.size(), fake.type)
                img_batch = Variable(img_batch, requires_grad = False)
                img_fper = Variable(img_fper, requires_grad = False)
                loss_d = F.mse_loss(self.discriminator(img_batch), img_fper)
                print("[", str(self.a_counter), "/", str(self.aprx_epochs), "] \t loss_d: ", loss_d)

                
                # Discriminator update
                self.loss += loss_d.item()
                loss_d.backward()
                self.optimizer.step()
                self.a_counter += 1
                #print("discriminator updated")

                disc_res = self.discriminator(fake)
                half_ideal = torch.ones_like(disc_res)
                ideal = torch.add(half_ideal, 21)
                loss_g = F.mse_loss(ideal, disc_res)
                
                return 0.0001*loss_g
            
            else:
                disc_res = self.discriminator(fake)
                half_ideal = torch.ones_like(disc_res)
                ideal = torch.add(half_ideal, 21)
                loss_g = F.mse_loss(ideal, disc_res)
                
                return torch.sum(loss_g)
    
    def state_dict(self, *args, **kwargs):
        state_discriminator = self.discriminator.state_dict(*args, **kwargs)
        state_optimizer = self.optimizer.state_dict()

        return dict(**state_discriminator, **state_optimizer)
               
# Some references
# https://github.com/kuc2477/pytorch-wgan-gp/blob/master/model.py
# OR
# https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py
