import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.fc_mu = nn.Linear(16 * 16 * 32, latent_dims)
        self.fc_logvar = nn.Linear(16 * 16 * 32, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        sigma = torch.exp(0.5 * logvar)
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = 0.5 * torch.sum(logvar.exp() + mu.pow(2) - 1 - logvar)
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dims, 16 * 16 * 32)
        self.convT1 = nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1)
        self.convT2 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)
        self.convT3 = nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1)

    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = x.view(z.size(0), 32, 16, 16)
        x = F.relu(self.convT1(x))
        x = F.relu(self.convT2(x))
        x = torch.sigmoid(self.convT3(x))
        return x

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)