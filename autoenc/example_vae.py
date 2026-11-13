import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VariationalAutoencoder(nn.Module):
   def __init__(self):
       super().__init__()
       self.mu = nn.Linear(32, 2)
       self.gamma = nn.Linear(32, 2)
       self.encoder = nn.Sequential(nn.Linear(784, 32), nn.ReLU(True))
       self.decoder = nn.Sequential(nn.Linear(2, 32), nn.ReLU(True), nn.Linear(32, 784), nn.Sigmoid())

   def forward(self, x):
       mu, gamma = self.encode(x)
       encoding = self.reparameterize(mu, gamma)
       x = self.decoder(encoding)
       return x, mu, gamma

   def reparameterize(self, mu, gamma):
       if self.training:
           sigma = torch.exp(0.5*gamma)
           std_z = torch.autograd.Variable(torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float())
           encoding = std_z.mul(sigma).add(mu)
           return encoding
       else:
           return mu

   def encode(self, x):
       x = self.encoder(x)
       mu = self.mu(x)
       gamma = self.gamma(x)
       return mu, gamma
 
   def decode(self, x):
       return self.decoder(x)

   def latent(self, x):
       mu, gamma = self.encode(x)
       encoding = self.reparameterize(mu, gamma)
       return encoding

def loss_function(input, output, mu, gamma, batch_size=batch_size):
   BCE = F.binary_cross_entropy(output, input)
   KLD = -0.5*torch.sum(1 + gamma - mu.pow(2) - gamma.exp())
   KLD /= batch_size*784
   return BCE + KLD
