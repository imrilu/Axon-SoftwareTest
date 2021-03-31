import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from MNISTDataModule import MNISTDataModule


class NetPL(pl.LightningModule):
    def __init__(self):
        super(NetPL, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.lr = 1.0

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def configure_optimizers(self):
        optimizer = optim.Adadelta(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        data, target = batch
        self.configure_optimizers().zero_grad()
        output = self(data)
        loss = F.nll_loss(output, target)
        # loss.backward()
        self.log('train_loss', loss)
        return loss

    def validation_Step(self, batch, batch_idx):
        data, target = batch
        self.configure_optimizers().zero_grad()
        output = self(data)
        val_loss = F.nll_loss(output, target)
        # loss.backward()
        self.log('val_loss', val_loss)
        return val_loss

    def test_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        test_loss = F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
        self.log('test_loss', test_loss)
        return test_loss


def main():
    dm = MNISTDataModule()
    model = NetPL()
    trainer = pl.Trainer(max_epochs=dm.args.epochs)
    trainer.fit(model, dm)
    if dm.args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
