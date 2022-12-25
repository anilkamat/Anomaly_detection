import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import scipy
from scipy import stats
from scipy.stats import norm

# read data
fileName = r'C:\Users\_Kamat_\Desktop\RPI\ResearchWork\Papers_\GNN\pythonCodes\Anamoly_detection\realAWSCloudwatch\ec2_cpu_utilization_5f5533.csv'
data = pd.read_csv(fileName)
data_noAnamoly = data.value[1:2930]  # data in tabular format
data_Anamoly = data.value[2930:]
plt.figure()
plt.plot(data_noAnamoly)
plt.plot(data_Anamoly)
plt.xlabel('Time stamps')
plt.xlabel('CPU usage')

def split_train_test(data_noAnamoly, data_Anamoly,window_size=50):
    mean, std = np.mean(data_Anamoly), np.std(data_Anamoly)
    data_Anamoly = (data_Anamoly-mean)/std
    n = data_Anamoly.shape[0] -window_size
    sequences_1 = []  # for anamoly
    for i in range(n):
        sequences_1.append(data_Anamoly[i:i+window_size])
    sequences_1 = torch.from_numpy(np.stack(sequences_1))
    
    mean, std = np.mean(data_noAnamoly), np.std(data_noAnamoly)
    data_noAnamoly = (data_noAnamoly-mean)/std
    n = data_noAnamoly.shape[0] -window_size
    sequences_2 = []      # for anamoly
    for i in range(n):
        sequences_2.append(data_noAnamoly[i:i+window_size])
    sequences_2 = torch.from_numpy(np.stack(sequences_2))
    return sequences_2, sequences_1

data_noAnamoly,data_Anamoly= split_train_test(data_noAnamoly, data_Anamoly) # data in window sequence format

X_train = Y_train = data_noAnamoly
X_test = Y_test = data_Anamoly


# Model 
input_size = X_train.shape[1]
h1_size = h22_size = 25
h3_size = 10
h2_size = h11_size = 20
output_size = Y_train.shape[1]
ELU = torch.nn.ELU(alpha = 0.5)


class MLP_model(nn.Module):
    def __init__(self, input_size, h1_size, h2_size, h3_size, h11_size,h22_size,output_size):
        super(MLP_model, self).__init__()
        self.input_size = input_size
        self.h1_size = h1_size
        self.h2_size = h2_size
        self.h3_size = h3_size   # bottle neck layer
        self.h11_size = h11_size
        self.h22_size = h22_size
        self.output_size = output_size

        self.L1 = nn.Linear(self.input_size, self.h1_size)
        self.L2 = nn.Linear(self.h1_size, self.h2_size)
        self.L3 = nn.Linear(self.h2_size, self.h3_size)
        self.L11 = nn.Linear(self.h3_size, self.h11_size)
        self.L22 = nn.Linear(self.h11_size, self.h22_size)
        self.Lout = nn.Linear(self.h22_size, self.output_size)

    def forward(self, input_X):
        L1_out = self.L1(input_X)
        L1_out = ELU(L1_out)
        L2_out = self.L2(L1_out)
        L2_out = ELU(L2_out)
        L3_out = self.L3(L2_out)
        L3_out = ELU(L3_out)
        L11_out = self.L11(L3_out)
        L11_out = ELU(L11_out)
        L22_out = self.L22(L11_out)
        L22_out = ELU(L22_out)
        Output = self.Lout(L22_out)
        return Output

model = MLP_model(input_size, h1_size, h2_size, h3_size, h11_size,h22_size,output_size).float()
optimizer = optim.Adam(model.parameters(), lr=0.05)
out_temp = model(X_train[0,:].float())

# train 
epochs = 500 # 200
MSE_loss = nn.MSELoss()
losses = []
losses_val = []
x_val = []
loss_last_epoch = []
print('training ...')
for i in range(epochs):
    output = model(X_train[0:-100,:].float())
    loss = MSE_loss(Y_train[0:-100,:].float(),output)
    losses.append(loss.detach().numpy())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i == epochs-1:
        for j in range(output.shape[0]):
            loss_last_epoch.append(MSE_loss(Y_train[j,:].float(),output[j,:]).detach().numpy()) 
    #validation
    if i%10 == 0: 
        output_val = model(X_test[-100:,:].float())
        loss_val = MSE_loss(output_val,Y_test[-100:,:])
        losses_val.append(loss_val.detach().numpy())
        x_val.append(i)

plt.figure()
# n = len(losses)
plt.plot(losses)
plt.plot(x_val, losses_val)
plt.title('convergence history')
plt.legend(['Train','Validataion'])

loss_last_epoch = np.array(loss_last_epoch)
mean = np.mean(loss_last_epoch)
std = np.std(loss_last_epoch)
pdf = stats.norm.pdf(loss_last_epoch, mean, std)

plt.figure()
n, bins, patches = plt.hist(loss_last_epoch, bins = 50)
bin_centers = 0.5*(bins[1:] + bins[:-1])
yy = ((1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-0.5 * (1 / std * (bins - mean))**2))
plt.plot(bins, yy, '--')
plt.title('histogram of train error')
xlim = plt.xlim

kde = stats.gaussian_kde(loss_last_epoch)
xx = np.linspace(0, 2, 50)
fig, ax = plt.subplots(figsize=(8,6))
ax.hist(loss_last_epoch, density=True, bins=50, alpha=0.5)
ax.plot(xx, kde(xx), linewidth = 2)

#test
print('testing ...')
output_test = model(X_test.float())
loss_test = MSE_loss(output_test, Y_test)
print(f"Total test loss {loss_test}")
plt.figure()
plt.plot(output_test[50,:].data)
plt.plot(Y_test[50,:].data)
plt.title('fitting')
plt.legend(['Model_prediction','Truth'])

threshold = mean+2*std
print(f'threshold (mean +2*std): {threshold} \n')
loss_temp = 0.0
num_anamoly = 0
for j in range(output_test.shape[0]):
    loss_temp = MSE_loss(Y_test[j,:].float(),output_test[j,:]).detach().numpy()
    if loss_temp > threshold:
        num_anamoly +=1 
print(f'total samples: {output_test.shape[0]} \n number of anamoly: {num_anamoly}')
