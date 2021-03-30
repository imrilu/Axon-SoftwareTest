import pytorch_lightning as pl
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

class MNISTDataModule(pl.LightningDataModule):

    def __init__(self):
        super().__init__()
        # Training settings
        parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
        parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                            help='input batch size for training (default: 64)')
        parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                            help='input batch size for testing (default: 1000)')
        parser.add_argument('--epochs', type=int, default=14, metavar='N',
                            help='number of epochs to train (default: 14)')
        parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                            help='learning rate (default: 1.0)')
        parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                            help='Learning rate step gamma (default: 0.7)')
        parser.add_argument('--no-cuda', action='store_true', default=False,
                            help='disables CUDA training')
        parser.add_argument('--dry-run', action='store_true', default=False,
                            help='quickly check a single pass')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                            help='how many batches to wait before logging training status')
        parser.add_argument('--save-model', action='store_true', default=False,
                            help='For Saving the current Model')
        self.args = parser.parse_args()
        self.use_cuda = not self.args.no_cuda and torch.cuda.is_available()

        torch.manual_seed(self.args.seed)

        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.train_kwargs = {'batch_size': self.args.batch_size}
        self.test_kwargs = {'batch_size': self.args.test_batch_size}
        if self.use_cuda:
            cuda_kwargs = {'num_workers': 1,
                           'pin_memory': True,
                           'shuffle': True}
            self.train_kwargs.update(cuda_kwargs)
            self.test_kwargs.update(cuda_kwargs)

    def prepare_data(self):
        mnist_train = datasets.MNIST('../data', train=True, download=True)
        mnist_test = datasets.MNIST('../data', train=False)

    def setup(self, stage):
        # transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        self.mnist_train = datasets.MNIST('../data', train=True, download=True, transform=transform)
        self.mnist_test = datasets.MNIST('../data', train=False, transform=transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.mnist_train, **self.train_kwargs)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.mnist_test, **self.test_kwargs)

