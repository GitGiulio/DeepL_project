import numpy as np
#import matplotlib.pyplot as plt
import sklearn
import torch
from torch import nn
from mlxtend.data import mnist_data

"""HYPER PARAMETERS"""
learning_rate = 0.001
test_size = 0.3
"""HYPER PARAMETERS"""

X,y = mnist_data()

Xtrain, Xtest, ytrain, ytest = sklearn.model_selection.train_test_split(X, y, test_size=test_size,shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

model = nn.Sequential(
    nn.Conv(20, 30),
    nn.ReLU(),
    nn.Linear(30, 4),
    nn.Softmax(1),
)

model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = nn.functional.cross_entropy

randomImput = torch.randn((3, 20))

randomImput.to(device)

print(randomImput)

print(model(randomImput))