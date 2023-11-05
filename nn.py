import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time

import random

# parse data
input_array = np.array(pd.read_csv("big_sound.csv"), dtype=np.float32)
print(input_array.shape)
output_array = np.array(pd.read_csv("big_data.csv"), dtype=np.float32)
#output_array = np.ones((505, 64), dtype=np.float32)*5
print(output_array.shape)

device = torch.device("gpu" if torch.cuda.is_available() else "cpu")

checkpoint_folder = "checkpoints_2"

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_array.shape[1], 50), # First fully connected layer
            #nn.ReLU(),
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, 75),
            #nn.ReLU(),
            nn.Linear(75, 75),
            nn.Softmax(),
            nn.Linear(75, output_array.shape[1])  # Second fully connected layer
        )

    def forward(self, input):
        return self.main(input)

"""
net = Net()


criterion = nn.L1Loss()

# setup optimizer
optimizer = optim.Adam(net.parameters(), lr=0.00002) # previously was lr=0.000002


# main loop
niter = 10000 # 10000
loss_data = []
for epoch in range(niter):
    np.random.shuffle(input_array)
    np.random.shuffle(output_array)
    for i in range(input_array.shape[0]):
        inputs = torch.from_numpy(input_array[i])
        labels = torch.from_numpy(output_array[i])
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print('[%d/%d][%d/%d] Loss: %.4f'
              % (epoch, niter, i, input_array.shape[0], loss.item()))
        loss_data.append(float(loss.item()))
    # do checkpointing
    torch.save(net.state_dict(), '%s/netG_epoch_%d.pth' % (checkpoint_folder, epoch))

plt.figure(figsize=(10, 6))
plt.plot(loss_data)
plt.xlabel('time')
plt.ylabel('loss')
plt.title('loss over time')
plt.grid(True)
plt.show()

"""

"""
previous attempt below
"""

"""
input_size = 5
hidden1_size = input_size
hidden2_size = input_size
hidden3_size = input_size
output_size = input_size + 1

net = nn.Sequential(
	nn.Linear(input_size, hidden1_size),
	nn.ReLU(),
	nn.Linear(hidden1_size, hidden2_size),
	nn.ReLU(),
	nn.Linear(hidden2_size, hidden3_size),
	nn.ReLU(),
	nn.Linear(hidden3_size, output_size)
	).to(device)

print(net)
"""

"""

# Import required libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Define the model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Load the dataset
dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
X = dataset[:,0:8]
Y = dataset[:,8]

# Fit the model
model.fit(X, Y, epochs=500, batch_size=10)

# Evaluate the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

"""



"""
from tinygrad.tensor import Tenso

from random import randint

def get_string_data():
	a = randint(0, 999)
	b = randint(0, 999)
	return (str(a)+"+"+str(b), str(a+b))

"""