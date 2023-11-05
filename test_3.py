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

checkpoint_folder = "checkpoints"

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


net = Net()
net.load_state_dict(torch.load("C:\\Users\\Adam\\Desktop\\personal projects\\0_mcgill_hackathon_2023\\Simulated-Auditory-Evoked-Hemodynamics\\checkpoints_2\\netG_epoch_100.pth"))
net.eval()

samples = input_array[np.random.choice(input_array.shape[0], size=5, replace=False)]

predictions = np.zeros((samples.shape[0], 71))

with torch.no_grad():
    for i, sample in enumerate(samples):
        newTensor = torch.from_numpy(sample)
        predictions[i] = net(newTensor)

print(predictions)
np.save("example_predictions", predictions)
"""
criterion = nn.MSELoss()

# setup optimizer
optimizer = optim.Adam(net.parameters(), lr=0.000002)

# main loop
niter = 8 # 10000
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
    #torch.save(net.state_dict(), '%s/netG_epoch_%d.pth' % (checkpoint_folder, epoch))

plt.figure(figsize=(10, 6))
plt.plot(loss_data)
plt.xlabel('time')
plt.ylabel('loss')
plt.title('loss over time')
plt.grid(True)
plt.show()
"""