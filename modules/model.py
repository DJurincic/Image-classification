import torch
from torch import nn
import torchvision
from torchvision.datasets import MNIST
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class ConvolutionalModel(nn.Module):
    """
    A PyTorch model representing a convolutional neural network.

    Args:
    -----

    in_channels (int): number of channels of input images
    conv1_width (int): the dimension of the convolution kernel in the first convolutional layer
    conv2_width (int): the dimension of the convolution kernel in the second convolutional layer
    fc1_width (int): the dimension of the fully-connected layer
    class_count (int): the number of output classes
    """

    def __init__(self, in_channels, conv1_width, conv2_width, fc1_width, class_count):

        super(ConvolutionalModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, conv1_width, kernel_size=5, stride=1, padding=2, bias=True)
        self.maxpool1=nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2=nn.Conv2d(conv1_width, conv2_width, kernel_size=5, stride=1, padding=2, bias=True)
        self.maxpool2=nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(conv2_width*7*7, fc1_width, bias=True)
        self.fc_logits = nn.Linear(fc1_width, class_count, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        """
        A function which initializes network weights in order to prevent problems while learning.
        """
        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear) and m is not self.fc_logits:
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

        self.fc_logits.reset_parameters()


    def forward(self, x):
        """
        A function which describes the forward pass through the network.

        Parameters:
        -----------

        x : torch Tensor
            Input image tensor
        """

        h = self.conv1(x)
        h=self.maxpool1(h)
        h = torch.relu(h)

        h=self.conv2(h)
        h=self.maxpool2(h)
        h=torch.relu(h)

        h = h.view(h.shape[0], -1)
        h = self.fc1(h)
        h = torch.relu(h)

        logits = self.fc_logits(h)


        return logits

def eval(model, dataloader, criterion):
    """
    A function used to evaluate the performance of a model, without calculating gradients and training.

    Parameters:
    -----------

    model : PyTorch Model
            A model which is being evaluated.
    dataloader : PyTorch DataLoader
            A data loader of evaluation data.
    criterion : PyTorch Loss
            A function for calculating the loss.

    Returns:
    --------

    eval_loss : the calculated loss on the evaluation set.
    eval_acc : the calculated accuracy on the evaluation set.
    """

    # Putting the model in eval mode and disableing gradient calculation.

    model.eval()

    with torch.no_grad():
        eval_loss = []
        correct = 0
        count = 0

        for eval_batch in dataloader:
            x, y = eval_batch

            x=torch.tensor(x)
            y_oh=torch.tensor(dense_to_one_hot(y,10))
        
        
            logits = model(x)
            loss = criterion(logits, y_oh)
            eval_loss.append(loss)

            # Calculating the number of correct classifications and counting the total number of examples for further calculations.
            correct += (logits.argmax(dim=-1) == y).float().sum()
            count += len(y)

        eval_loss = torch.mean(torch.tensor(eval_loss))
        eval_acc = correct / count
  

        return eval_loss.item(), eval_acc.item()


def train(model, dataloader, criterion, optimizer):

    """
    A function used to evaluate the performance of a model, without calculating gradients and training.

    Parameters:
    -----------

    model : PyTorch Model
            A model which is being evaluated.
    dataloader : PyTorch DataLoader
            A data loader of evaluation data.
    criterion : PyTorch Loss
            A function for calculating the loss.
    optimizer : PyTorch Optimizer
            An optimizer used to update model weights based on the loss.

    Returns:
    --------

    The average loss on the training set.
    """

    # Putting the model in train mode.

    model.train()

    train_loss=[]

    for x,y in dataloader:

        x=torch.tensor(x)
        y=torch.tensor(dense_to_one_hot(y,10))

        logits=model(x)
        loss=criterion(logits,y)
        train_loss.append(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return (sum(train_loss)/len(train_loss)).item()


def dense_to_one_hot(y, class_count):
    return np.eye(class_count)[y]