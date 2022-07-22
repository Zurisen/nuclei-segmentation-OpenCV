import os
import cv2
import numpy as np
import glob
import PIL.Image as Image
import pandas as pd
# pip install torchsummary
import torch
import torch.nn as nn
from tqdm.notebook import tqdm
from torchvision import models
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
#from torchsummary import summary
import torch.optim as optim
import torchvision.transforms.functional as TF
from skimage import data, io, img_as_float, exposure
import matplotlib.pyplot as plt

#Dataset Loader
class LoadDataSet(torch.utils.data.Dataset):
        def __init__(self, path, train=True, IMG_HEIGHT=256, IMG_WIDTH=256, rotation=0):
            self.path = path
            self.IMG_HEIGHT = IMG_HEIGHT
            self.IMG_WIDTH = IMG_WIDTH
            self.folders = os.listdir(path)
            self.rotation = rotation
            train_indx = int(0.95*len(self.folders))
            if train:
                self.folders = self.folders[:train_indx]
            else:
                self.folders = self.folders[train_indx:]
            
            self.transforms = transforms.Compose(
                [transforms.Resize((self.IMG_HEIGHT, self.IMG_WIDTH)), 
                                    transforms.ToTensor()])
        
        def __len__(self):
            return len(self.folders)
              
        
        def __getitem__(self,idx):
            image_folder = os.path.join(self.path,
                                        self.folders[idx],'images/')
            mask_folder = os.path.join(self.path,
                                       self.folders[idx],'masks/')
            image_path = os.path.join(image_folder,
                                      os.listdir(image_folder)[0])
            
            img = Image.open(image_path)
            img = self.transforms(img)
            img = TF.rotate(img, self.rotation)
            
            mask = self.get_mask(mask_folder)
            mask = torch.from_numpy(mask).reshape(1,self.IMG_HEIGHT,self.IMG_WIDTH)
            return img, mask


        def get_mask(self,mask_folder):
            mask = np.zeros((self.IMG_HEIGHT, self.IMG_WIDTH, 1),
                            dtype=np.bool)
            for mask_ in os.listdir(mask_folder):
                    mask_ = Image.open(os.path.join(mask_folder,mask_))
                    mask_ = self.transforms(mask_)
                    mask_ = TF.rotate(mask_, self.rotation).numpy()
                    mask_ = np.expand_dims(mask_,axis=-1)
                    mask = np.maximum(mask, mask_)
              
            return mask

def plot_img_and_hist(image, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram.

    """
    image = img_as_float(image)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf