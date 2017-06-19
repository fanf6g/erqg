import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable

from dataset_gen2 import DatasetGenerator2


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        D_in, H, D_out = 200, 180, 2
        self.line1 = nn.Linear(D_in, H)
        self.relu = nn.ReLU()
        self.line2 = nn.Linear(H, D_out)

        self.criterion = nn.CrossEntropyLoss(weight=[1, 1])
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.line1(x)
        x = self.relu(x)
        x = self.line2(x)
        return x

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

        x_train = Variable(torch.from_numpy(np.array(x_train))).float()
        y_train = Variable(torch.from_numpy(np.array(y_train)), requires_grad=False).long()

        x_test = Variable(torch.from_numpy(np.array(x_test))).float()
        y_test = Variable(torch.from_numpy(np.array(y_test)), requires_grad=False).long()

        return x_train, y_train, x_test, y_test

    def evaluate(self, x_test, y_test, net):
        pred_test = net.forward(x_test)

        _, predicted = torch.max(pred_test.data, 1)

        match_pred = set(predicted.numpy().T[0].nonzero()[0])
        match_y = set(y_test.data.numpy().nonzero()[0])
        tp = match_y.intersection(match_pred)
        print('tp:{0}, found:{1}, precision: {2}, recall: {3}'.format(len(tp), len(match_pred),
                                                                      len(tp) / len(match_pred),
                                                                      len(tp) / len(match_y)))

    def onerow(self, n, net):
        dg2 = DatasetGenerator2()
        features, labels = dg2.onerow(n)
        x_test = Variable(torch.from_numpy(np.array(features))).float()
        y_test = Variable(torch.from_numpy(np.array(labels)), requires_grad=False).long()

        pred_test = net.forward(x_test)
        _, predicted = torch.max(pred_test.data, 1)

        match_pred = set(predicted.numpy().T[0].nonzero()[0])
        match_y = set(y_test.data.numpy().nonzero()[0])
        tp = match_y.intersection(match_pred)
        print('tp:{0}, found:{1}, precision: {2}, recall: {3}'.format(len(tp), len(match_pred),
                                                                      len(tp) / len(match_pred),
                                                                      len(tp) / len(match_y)))

    def fit(self, x_train, y_train):
        for i in range(2000):  # loop over the dataset multiple times

            running_loss = 0.0

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net.forward(x_train)
            loss = self.criterion(outputs, y_train)
            loss.backward()
            self.optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % 200 == 199:  # print every 2000 mini-batches
                print('loss:{0}'.format(running_loss / 20))

                running_loss = 0.0

        print('Finished Training')


def fit(x_train, y_train, net):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for i in range(2000):  # loop over the dataset multiple times

        running_loss = 0.0

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net.forward(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 200 == 199:  # print every 2000 mini-batches
            print('loss:{0}'.format(running_loss / 20))

            running_loss = 0.0

    print('Finished Training')


if __name__ == "__main__":
    net = Net()
    loss = nn.CrossEntropyLoss()
    print(net)

    x_train, y_train, x_test, y_test = net.load()
    # net.fit(x_train, y_train)
    fit(x_train, y_train, net)

    # pred = net.forward(x_train)
    # print(pred.size())
    # print(pred)

    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # for i in range(2000):  # loop over the dataset multiple times
    #
    #     running_loss = 0.0
    #
    #     # zero the parameter gradients
    #     optimizer.zero_grad()
    #
    #     # forward + backward + optimize
    #     outputs = net.forward(x_train)
    #     loss = criterion(outputs, y_train)
    #     loss.backward()
    #     optimizer.step()
    #
    #     # print statistics
    #     running_loss += loss.data[0]
    #     if i % 200 == 199:  # print every 2000 mini-batches
    #         print('loss:{0}'.format(running_loss / 20))
    #
    #         running_loss = 0.0
    #
    # print('Finished Training')

    net.evaluate(x_test, y_test, net)
    net.onerow(6, net)



    # net.train()
    # data = torch.autograd.Variable(torch.randn(5))
    # print(data)
    # print(F.softmax(data))
    # print(F.softmax(data).sum())  # Sums to 1 because it is a distribution!
    # print(F.log_softmax(data))  # theres also log_softmax
