# -*- coding: utf-8 -*-
import pickle

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
from dataset_gen2 import DatasetGenerator2


class shn():
    '''
    Single Hidden Nuerual Network.
    '''

    def __init__(self):
        D_in, H, D_out = 200, 180, 2
        self.model = torch.nn.Sequential(
            torch.nn.Linear(D_in, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, D_out),
        )
        self.loss_fn = torch.nn.MSELoss(size_average=False)

        self.learning_rate = 1e-3
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def load(self, file='train'):
        n_train = 500
        with open(file, mode='rb') as f:
            pos, neg = pickle.load(f)

            train_pos, test_pos = pos[:n_train], pos[n_train:]
            train_neg, test_neg = neg[:n_train], neg[n_train:]

            train = []
            train.extend(train_pos)
            train.extend(train_neg)
            np.random.shuffle(train)

            test = []
            test.extend(test_pos)
            test.extend(test_neg)
            np.random.shuffle(test)

        train = pd.DataFrame(train)
        x_train = train[0].tolist()
        y_train = train[1].tolist()

        test = pd.DataFrame(test)
        x_test = test[0].tolist()
        y_test = test[1].tolist()

        # N, D_in, H, D_out = len(x_train), 200, 150, 2
        # N, D_in, H, D_out = len(x_train), 200, 150, 2

        tensor_y = np.zeros(shape=((len(x_train)), 2), dtype='float')
        tensor_y[list(range(len(x_train))), y_train] = 1.0

        x_train = Variable(torch.from_numpy(np.array(x_train))).float()
        y_train = Variable(torch.from_numpy(np.array(tensor_y)), requires_grad=False).float()

        x_test = Variable(torch.from_numpy(np.array(x_test))).float()
        y_test = torch.from_numpy(np.array(y_test))

        return x_train, y_train, x_test, y_test

    def training(self, x_train, y_train, savePath=None):
        # Create random Tensors to hold inputs and outputs, and wrap them in Variables.


        for t in range(1000):
            # Forward pass: compute predicted y by passing x to the model.

            y_pred = self.model(x_train)

            # Compute and print loss.
            loss = self.loss_fn(y_pred, y_train)
            print(t, loss.data[0])

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable weights
            # of the model)
            self.optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            loss.backward()

            # Calling the step function on an Optimizer makes an update to its
            # parameters
            self.optimizer.step()

        if savePath is not None:
            with open(savePath, mode='wb') as f:
                pickle.dump(self.model, f)
            pass

    def evaluate(self, x_test, y_test):
        correct = 0
        total = len(y_test)

        print(x_test)
        print(y_test)

        y_pred = self.model(x_test)
        print(y_pred)
        _, predicted = torch.max(y_pred.data, 1)

        print(predicted)

        correct += (predicted == y_test).sum()
        print(correct)

        print('Accuracy of the network on the 100 test images: {0}'.format(100.0 * correct / total))

    def recover(self):
        dg = DatasetGenerator2()
        x_test, y_test = dg.onerow(3)

        x_test = Variable(torch.from_numpy(np.array(x_test))).float()
        y_test = torch.from_numpy(np.array(y_test)).long()

        y_pred = self.model(x_test).int()
        print(y_pred)
        _, predicted = torch.max(y_pred.data, 1)

        print(predicted)
        total = len(x_test)
        correct = 0
        correct += (predicted == y_test).sum()
        print(correct)

        print('Accuracy of the network on the 100 test images: {0}'.format(100.0 * correct / total))


if __name__ == "__main__":
    model = shn()
    x_train, y_train, x_test, y_test = model.load()
    model.training(x_train, y_train)
    # model.evaluate(x_test, y_test)
    model.recover()
    # dg2 = DatasetGenerator2()
    # features,labels = dg2.onerow(3)
    # print(labels)
    # it = model.recover()
    # a = it.__next__()
    # print(a.shape)
    # print(a)
    # print(a.sum())
    pass
