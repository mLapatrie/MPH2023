import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("gpu" if torch.cuda.is_available() else "cpu")

checkpoint_folder = "checkpoints"

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(16, 500), # First fully connected layer
            nn.ReLU(),
            nn.Linear(500, 148)  # Second fully connected layer
        )

    def forward(self, input):
        return self.main(input)


net = Net()


criterion = nn.BCELoss()

# setup optimizer
optimizer = optim.Adam(net.parameters(), lr=0.005)

# main loop
niter = 100
for epoch in range(niter):
    for i, data in enumerate(dataset):
        inuts, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print('[%d/%d][%d/%d] Loss: %.4f'
              % (epoch, niter, i, len(dataset), loss.item()))
    # do checkpointing
    torch.save(net.state_dict(), '%s/netG_epoch_%d.pth' % (checkpoint_folder, epoch))

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