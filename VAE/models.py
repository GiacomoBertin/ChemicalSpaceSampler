import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F


class Hybrid_VAE_FEE(nn.Module):
    def __init__(self, embed_dimension, device=None):
        super(Hybrid_VAE_FEE, self).__init__()
        self.conv1d1 = nn.Conv1d(120, 9, kernel_size=9)
        self.conv1d2 = nn.Conv1d(9, 9, kernel_size=9)
        self.conv1d3 = nn.Conv1d(9, 10, kernel_size=11)
        self.fc0 = nn.Linear(90, 435)
        self.fc11 = nn.Linear(435, embed_dimension)
        self.fc12 = nn.Linear(435, embed_dimension)
        self.fcee0 = nn.Linear(embed_dimension, embed_dimension - 1)
        self.fcee1 = nn.Linear(embed_dimension - 1, 1)

        self.fc2 = nn.Linear(embed_dimension, embed_dimension)
        self.gru = nn.GRU(embed_dimension, 501, 3, batch_first=True)
        self.fc3 = nn.Linear(501, 35)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.to(self.device)

    def encode(self, x):
        h = F.relu(self.conv1d1(x))
        h = F.relu(self.conv1d2(h))
        h = F.relu(self.conv1d3(h))
        h = h.view(h.size(0), -1)
        h = F.selu(self.fc0(h))
        return self.fc11(h), self.fc12(h)

    def free_energy_estimation(self, x):
        h = F.relu(self.fcee0(x))
        return self.fcee1(h)

    def reparametrize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = 1e-2 * torch.randn_like(std)
            w = eps.mul(std).add_(mu)
            return w
        else:
            return mu

    def decode(self, z):
        z = F.selu(self.fc2(z))
        z = z.view(z.size(0), 1, z.size(-1)).repeat(1, 120, 1)
        out, h = self.gru(z)
        out_reshape = out.contiguous().view(-1, out.size(-1))
        y0 = F.softmax(self.fc3(out_reshape), dim=1)
        y = y0.contiguous().view(out.size(0), -1, y0.size(-1))
        return y

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        dg = self.free_energy_estimation(z)
        return self.decode(z), mu, logvar, dg


class MolecularFEE(nn.Module):
    def __init__(self, embed_dimension, device=None):
        super(MolecularFEE, self).__init__()
        self.conv1d1 = nn.Conv1d(120, 9, kernel_size=9)
        self.conv1d2 = nn.Conv1d(9, 9, kernel_size=9)
        self.conv1d3 = nn.Conv1d(9, 10, kernel_size=11)
        self.fc0 = nn.Linear(90, 435)
        self.fc1 = nn.Linear(435, embed_dimension)
        self.fc2 = nn.Linear(embed_dimension, 1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.to(self.device)

    def forward(self, x):
        h = F.relu(self.conv1d1(x))
        h = F.relu(self.conv1d2(h))
        h = F.relu(self.conv1d3(h))
        h = h.view(h.size(0), -1)
        h = F.selu(self.fc0(h))
        h = F.selu(self.fc1(h))
        return self.fc2(h)


class MolecularVAE(nn.Module):
    def __init__(self, embed_dimension, device=None):
        super(MolecularVAE, self).__init__()

        self.conv1d1 = nn.Conv1d(120, 9, kernel_size=9)
        self.conv1d2 = nn.Conv1d(9, 9, kernel_size=9)
        self.conv1d3 = nn.Conv1d(9, 10, kernel_size=11)
        self.fc0 = nn.Linear(90, 435)
        self.fc11 = nn.Linear(435, embed_dimension)
        self.fc12 = nn.Linear(435, embed_dimension)

        self.fc2 = nn.Linear(embed_dimension, embed_dimension)
        self.gru = nn.GRU(embed_dimension, 501, 3, batch_first=True)
        self.fc3 = nn.Linear(501, 35)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.to(self.device)

    def encode(self, x):
        h = F.relu(self.conv1d1(x))
        h = F.relu(self.conv1d2(h))
        h = F.relu(self.conv1d3(h))
        h = h.view(h.size(0), -1)
        h = F.selu(self.fc0(h))
        return self.fc11(h), self.fc12(h)

    def reparametrize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = 1e-2 * torch.randn_like(std)
            w = eps.mul(std).add_(mu)
            return w
        else:
            return mu

    def decode(self, z):
        z = F.selu(self.fc2(z))
        z = z.view(z.size(0), 1, z.size(-1)).repeat(1, 120, 1)
        out, h = self.gru(z)
        out_reshape = out.contiguous().view(-1, out.size(-1))
        y0 = F.softmax(self.fc3(out_reshape), dim=1)
        y = y0.contiguous().view(out.size(0), -1, y0.size(-1))
        return y

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar
