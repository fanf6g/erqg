# -*- coding: utf-8 -*-
import pickle

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.

file = 'train'
with open(file, mode='rb') as f:
    pos, neg = pickle.load(f)

    train = []
    train.extend(pos[:100])
    train.extend(neg[:220])
    np.random.shuffle(train)

    test = []
    test.extend(pos[100:])
    test.extend(neg[220:])

# print(train)

train = pd.DataFrame(train)
x_train = train[0].tolist()
y_train = train[1].tolist()

N, D_in, H, D_out = 320, 200, 150, 2

# idx = np.array(list(zip(range(N), y)))

tensor_y = np.zeros(shape=(N, D_out), dtype='float')
tensor_y[list(range(N)), y_train] = 1.0

# Create random Tensors to hold inputs and outputs, and wrap them in Variables.

x_train = Variable(torch.from_numpy(np.array(x_train))).float()
y_train = Variable(torch.from_numpy(np.array(tensor_y)), requires_grad=False).float()
print(x_train)
print(y_train)

# Use the nn package to define our model and loss function.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)
torch.nn.NLLLoss(weight=[3, 1])
loss_fn = torch.nn.MSELoss(size_average=False)

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algoriths. The first argument to the Adam constructor tells the
# optimizer which Variables it should update.

# datasets = DatasetGenerator()

# batch = datasets.next_batch(N)



learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(1000):
    # Forward pass: compute predicted y by passing x to the model.

    y_pred = model(x_train)

    # Compute and print loss.
    loss = loss_fn(y_pred, y_train)
    print(t, loss.data[0])

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable weights
    # of the model)
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()

correct = 0
total = 200

test = pd.DataFrame(test)
x_test = test[0].tolist()
y_test = test[1].tolist()

N_test = 200
# idx = np.array(list(zip(range(N), y)))
#
test_y = np.zeros(shape=(N_test, D_out), dtype='float')
# test_y[list(range(N_test)), y_test] = 1.0

x_test = Variable(torch.from_numpy(np.array(x_test))).float()
y_test = torch.from_numpy(np.array(y_test))

print(x_test)
print(y_test)

y_pred = model(x_test)
print(y_pred)
_, predicted = torch.max(y_pred.data, 1)

print(predicted)

correct += (predicted == y_test).sum()
print(correct)

print('Accuracy of the network on the 100 test images: %d %%' % (100.0 * correct / total))
