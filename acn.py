"""
Associative Compression Network based on https://arxiv.org/pdf/1804.02476v2.pdf

This is a VAE with a conditional prior.
"""


import os.path as osp
import random

import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms

CODE_LEN = 20

batch_size = 128

FPATH_VAE = osp.join('models', 'vae.params')
FPATH_PRIOR = osp.join('models', 'prior.params')


data_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True)


class VAE(torch.nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2_u = nn.Linear(512, CODE_LEN)
        self.fc2_logstd = nn.Linear(512, CODE_LEN)
        self.fc3 = nn.Linear(CODE_LEN, 512)
        self.fc4 = nn.Linear(512, 28 * 28)

    def encode(self, inputs: torch.Tensor):

        h1 = F.relu(self.fc1(inputs))
        mu, logstd = self.fc2_u(h1), self.fc2_logstd(h1)

        return mu, logstd

    def decode(self, latent: torch.Tensor):

        h3 = F.relu(self.fc3(latent))
        h4 = F.sigmoid(self.fc4(h3))

        return h4

    def forward(self, inputs: torch.Tensor):

        u, logstd = self.encode(inputs)
        h2 = self.reparametrize(u, logstd)
        output = self.decode(h2)

        return output, u, logstd

    def reparametrize(self,
                      u: torch.Tensor,
                      s: torch.Tensor):
        """
        Draw from standard normal distribution (as input) and then parametrize it (so params can be backpropped).

        Args:
            u: Means.
            s: Log standard deviations.

        Returns: Draws from a distribution.
        """

        if self.training:
            std = s.exp()

            activations = u + std * torch.randn(u.shape)

            return activations

        return u


class PriorNetwork(torch.nn.Module):

    def __init__(self, k=5):
        """
        Args:
            k: Number of neighbors to choose from when picking code to condition prior.
        """

        super().__init__()

        self.fc1 = nn.Linear(CODE_LEN, 512)
        self.fc2_u = nn.Linear(512, CODE_LEN)
        self.fc2_s = nn.Linear(512, CODE_LEN)

        self.k = k
        self.knn = KNeighborsClassifier(n_neighbors=2 * k)

        codes = torch.randn((len(data_loader.dataset), CODE_LEN)).numpy()

        self.fit_knn(codes)

    def pick_close_neighbor(self, code: torch.Tensor) -> torch.Tensor:
        """
        K-nearest neighbors to choose a close code. This emulates an ordering of the original data.

        Args:
            code: Latent activations of current training example.

        Returns: Numpy array of same dimension as code.
        """

        # TODO(jalex): This is slow - can I make it faster by changing search algo/leaf size?
        neighbor_idxs = self.knn.kneighbors([code.detach().numpy()], return_distance=False)[0]

        valid_idxs = [n for n in neighbor_idxs if n not in self.seen]

        if len(valid_idxs) < self.k:

            codes_new = [c for i, c in enumerate(self.codes) if i not in self.seen]
            self.fit_knn(codes_new)

            return self.pick_close_neighbor(code)

        neighbor_codes = [self.codes[idx] for idx in valid_idxs]

        if len(neighbor_codes) > self.k:
            code_np = code.detach().numpy()
            # this is same metric KNN uses
            neighbor_codes = sorted(neighbor_codes, key=lambda n: ((code_np - n) ** 2).sum())[:self.k]

        neighbor = random.choice(neighbor_codes)

        return neighbor

    def fit_knn(self, codes: np.ndarray):
        """
        Reset the KNN. This can be used when we get too many misses or want to update the codes.

        Args:
            codes: New codes to fit.
        """

        self.codes = codes
        self.seen = set()

        y = [0] * len(codes)

        self.knn.fit(codes, y)

    def forward(self, codes: torch.Tensor):
        """
        Calculate prior conditioned on codes.

        Args:
            codes: latent activations.

        Returns: Two parameters each of dimensionality codes. These can be used as mu, std for a Gaussian.
        """

        # Can use this to emulate uncoditional prior
        # return torch.zeros(codes.shape[0], 1), torch.ones(codes.shape[0], 1)

        previous_codes = [self.pick_close_neighbor(c) for c in codes]
        previous_codes = torch.tensor(previous_codes)

        return self.encode(previous_codes)

    def encode(self, prev_code: torch.Tensor):

        h1 = F.relu(self.fc1(prev_code))
        mu, logstd = self.fc2_u(h1), self.fc2_s(h1)

        return mu, logstd


def calc_loss(x, recon, u_q, s_q, u_p, s_p):
    """
    Loss derived from variational lower bound (ELBO) or information theory (see bits-back for details).

    The loss comprises two parts:

    1. Reconstruction loss (how good the VAE is at reproducing the output).
    2. The coding cost (KL divergence between the model posterior and conditional prior).

    Args:
        x: Inputs.
        recon: Reconstruction from a VAE.
        u_q: Mean of model posterior.
        s_q: Log std of model posterior.
        u_p: Mean of (conditional) prior.
        s_p: Log std of (conditional) prior.

    Returns: Loss.
    """

    # reconstruction
    xent = F.binary_cross_entropy(recon, x, size_average=False)

    # coding cost
    dkl = torch.sum(s_p - s_q - 0.5 + ((2 * s_q).exp() + (u_q - u_p).pow(2)) / (2 * (2 * s_p).exp()))

    return xent + dkl


def train(vae: VAE,
          prior: PriorNetwork,
          optimizer: torch.optim.Optimizer,
          idx_epoch: int):

    vae.train()

    train_loss = 0

    new_codes = []

    for batch_idx, (data, _) in enumerate(data_loader):

        inputs = data.view(data.shape[0], -1)

        optimizer.zero_grad()

        outputs, u_q, s_q = vae(inputs)
        new_codes.append(u_q)

        u_p, s_p = prior(u_q)

        loss = calc_loss(inputs, outputs, u_q, s_q, u_p, s_p)

        loss.backward()
        train_loss += loss.item()

        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                idx_epoch, batch_idx * len(data), len(data_loader.dataset), 100 * batch_idx / len(data_loader),
                loss.item() / len(data)))

            torch.save(vae.state_dict(), FPATH_VAE)
            torch.save(prior.state_dict(), FPATH_PRIOR)

    new_codes = torch.cat(new_codes).detach().numpy()
    prior.fit_knn(new_codes)

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        idx_epoch, train_loss / len(data_loader.dataset)))


def daydream(image: torch.Tensor,
             vae: VAE,
             prior: PriorNetwork,
             num_iters: int=0,
             save_gif: bool=False):

    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig = plt.figure()
    plt.axis('off')

    mpl_images = []

    i = 0
    while i < num_iters or not num_iters:

        image_np = image.detach().numpy().reshape(28, 28)

        x = plt.imshow(image_np, cmap='Greys', animated=True)
        mpl_images.append([x])

        if not num_iters:
            plt.axis('off')
            plt.show()

        latent, _ = vae.encode(image)

        # Generate guess of next latent
        next_latent, _ = prior.encode(latent)

        image = vae.decode(next_latent)

        image = torch.tensor(image.reshape(1, -1))

        i += 1

    ani = animation.ArtistAnimation(fig, mpl_images, interval=200, blit=True)
    plt.show()

    if save_gif:
        ani.save('animation.gif', writer='imagemagick', fps=5)


def parse_flags():

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--task', default='train', choices=['train', 'daydream'], help='Task to do.')

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_flags()

    vae = VAE()
    prior = PriorNetwork()

    if args.task == 'train':

        params = list(vae.parameters()) + list(prior.parameters())
        optimizer = optim.Adam(params)

        for i in range(100):
            train(vae, prior, optimizer, i)
    else:

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True)

        img, _ = next(iter(test_loader))

        vae.load_state_dict(torch.load(FPATH_VAE))
        prior.load_state_dict(torch.load(FPATH_PRIOR))

        img_ = img[0].view(1, -1)

        daydream(img_, vae, prior, num_iters=0)
