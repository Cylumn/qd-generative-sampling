import torch.nn as nn
import torch.nn.functional as F
import torch

import fire
from tqdm import tqdm

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder

from dataset import ShapesDataset

class ShapeClassifier(nn.Module):
    def __init__(self):
        super(ShapeClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128*128, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

def eval(model):
    bias = 0.5
    size = 5000
    test_dataset = ShapesDataset(size=size, color_bias=bias, stable='data/eval.pt')

    red_triangle = []
    blue_triangle = []
    red_square = []
    blue_square = []
    with torch.no_grad():
        for i in tqdm(range(int(size/2 * bias))):
            x = test_dataset[i][0]
            red_triangle.append(model(x.to('cuda').unsqueeze(0)))

        for i in tqdm(range(int(size/2 * bias), size//2)):
            x = test_dataset[i][0]
            blue_triangle.append(model(x.to('cuda').unsqueeze(0)))

        for i in tqdm(range(size//2, int(size/2 * bias) + size//2)):
            x = test_dataset[i][0]
            blue_square.append(model(x.to('cuda').unsqueeze(0)))

        for i in tqdm(range(int(size/2 * bias) + size//2, size)):
            x = test_dataset[i][0]
            red_square.append(model(x.to('cuda').unsqueeze(0)))

        x = torch.tensor(blue_triangle)
        acc_bt = len(x[(x.round() == 0)]) / len(blue_triangle)

        x = torch.tensor(red_square)
        acc_rs = len(x[(x.round() == 1)]) / len(red_square)

        x = torch.tensor(red_triangle)
        acc_rt = len(x[(x.round() == 0)]) / len(red_triangle)

        x = torch.tensor(blue_square)
        acc_bs = len(x[(x.round() == 1)]) / len(blue_square)
    
    return (acc_bt + acc_rs + acc_rt + acc_bs) / 4
    
def run_trial(data_path,
              color_bias: float = 0.98):
    print(f"> color_bias = {color_bias}")

    real_dataset = ShapesDataset(size=1000, 
                                 color_bias=color_bias, 
                                 stable=f'data/train_b={color_bias}.pt')
    real_dataloader = DataLoader(real_dataset, batch_size=128, shuffle=True)
    qdgs_dataset = ImageFolder(f'data/{data_path}', transform=transforms.ToTensor())
    qdgs_dataloader = DataLoader(qdgs_dataset, batch_size=128, shuffle=True)

    model = ShapeClassifier().to('cuda')
    opt = torch.optim.Adam(model.parameters())
    crit = nn.BCELoss()
    transform_train = transforms.RandomRotation(180)
  
    ### TRAIN WITH BIASED DATA ###
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in tqdm(range(20)):
        epoch_loss = 0
        for x, y, _ in real_dataloader:
            x = x.to('cuda') # GPU
            opt.zero_grad()
            y_hat = model(transform_train(x))
            loss = crit(y_hat.squeeze(),y.float().to('cuda'))
            loss.backward()
            epoch_loss += loss.item()
            opt.step()

    ### FINE-TUNE WITH DEBIASED DATA ###
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in tqdm(range(30)):
        epoch_loss = 0
        for x, y in qdgs_dataloader:
            x = x.to('cuda') # GPU
            opt.zero_grad()
            y_hat = model(transform_train(x))
            loss = crit(y_hat.squeeze(),y.float().to('cuda'))
            loss.backward()
            epoch_loss += loss.item()
            opt.step()

    acc = eval(model)
    print("Fine-tuned accuracy:", acc)

if __name__ == '__main__':
    fire.Fire(run_trial)
