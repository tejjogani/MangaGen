import numpy as np
from PIL import Image
import glob
import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms

class AnimeDataset(Dataset):
    def __init__(self, folder_path):
        """
        Args:
            folder_path (string): path to image folder
        """
        # Get image list
        self.image_list = glob.glob(folder_path+'*')
        # Calculate len
        self.data_len = len(self.image_list)
        #transforms
        self.convert_image = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(), 
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])


    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_path = self.image_list[index]
        # Open image
        im_as_im = Image.open(single_image_path)
        # Do some operations on image
        # Convert to numpy, dim = 89x89
        im_as_ten = self.convert_image(im_as_im)
        return im_as_ten

    def __len__(self):
        return self.data_len

if __name__ == '__main__':
	dataset = AnimeDataset('images/')
	dataset.__getitem__(2)
