import os
import torch

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
    
class ShapesDataset(Dataset):
    def __init__(self, size=100, image_size=128, color_bias=0.98, stable=False):
        self.size = size
        self.image_size = image_size
        self.color_bias = color_bias
        self.data = self.generate_data()
        
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(180),     # Random rotation up to 45 degrees
            transforms.RandomResizedCrop(self.image_size, scale=(0.5, 2)),  # Random resizing with cropping to 64x64
            transforms.ToTensor(),             # Convert to tensor
        ])
        
        self.stable = False
        if type(stable) == str and os.path.exists(stable):
            self.samples = torch.load(stable)
        elif stable:
            self.samples = []
            for i in range(self.__len__()):
                self.samples.append(self.__getitem__(i)[0])
            self.samples = torch.stack(self.samples)

            torch.save(self.samples, stable)
        self.stable = stable
                
    
    def generate_data(self):
        # data = torch.zeros((self.size, 3, self.image_size, self.image_size), dtype=torch.float32)
        data = torch.zeros(4, 3, 128, 128)
        
        # Red Triangle
        size = 64
        for i in range(size):
            for j in range(abs(size//2-i), size//2):
                data[0, :, 64-size//2+j, 64-size//2+i] = torch.tensor([216, 27, 96]).div(255)
        
        # Blue Triangle
        for i in range(size):
            for j in range(abs(size//2-i), size//2):
                data[1, :, 64-size//2+j, 64-size//2+i] = torch.tensor([30, 136, 229]).div(255)
        
        # Red Square
        size = 36
        for i in range(size):
            for j in range(size):
                data[2, :, 64-size//2+j, 64-size//2+i] = torch.tensor([216, 27, 96]).div(255)
        
        # Blue Square
        for i in range(size):
            for j in range(size):
                data[3, :, 64-size//2+j, 64-size//2+i] = torch.tensor([30, 136, 229]).div(255)
                
        return data

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if idx < (self.size // 2) * self.color_bias:
            index = 0
            color = 0
        elif idx < (self.size // 2):
            index = 1
            color = 1
        elif idx < (self.size // 2) * self.color_bias + self.size // 2:
            index = 3
            color = 0
        elif idx < (self.size // 2) * 2:
            index = 2
            color = 1
        
        if self.stable:
            return self.samples[idx], index // 2, color
        else:
            sample = self.data[index]
            if self.transforms:
                sample = self.transforms(sample)

            black_pixels = (sample.lt(0.5)).all(dim=0)
            # Set the values of black pixels to white (255)
            sample[:, black_pixels] = 1

            return sample, index // 2, color