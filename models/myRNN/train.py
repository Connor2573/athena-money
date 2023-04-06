import torch
import os
import pandas as pd
from models import athenaLTSM
from torch.utils.data import DataLoader
from torch.autograd import Variable 
from myStockDataset import myData, loadAllMyCsvsIntoDataframeList, loadSingleCsv, preprocess, loaderTickerData
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

mm = MinMaxScaler()
ss = StandardScaler()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

num_epochs = 1000 #1000 epochs
learning_rate = 0.001 #0.001 lr

input_size = 7 #number of features
hidden_size = 128 #number of features in hidden state
num_layers = 1 #number of stacked lstm layers
seq_length = 10
num_classes = 1 #number of output classes 
split = 400 

#df = loadSingleCsv('AAPL2.csv')
df = loaderTickerData('CVNA')

X = df.iloc[:, 3:]
Y = df.iloc[:, 2:3]

X_ss = ss.fit_transform(X)
y_mm = mm.fit_transform(Y) 

print("Input shape: {}", X_ss.shape)
print("Labels shape: {}", y_mm.shape)

X_train = X_ss[:split, :]
X_test = X_ss[split:, :]

y_train = y_mm[:split, :]
y_test = y_mm[split:, :] 

print("Training Shape", X_train.shape, y_train.shape)
print("Testing Shape", X_test.shape, y_test.shape)

X_train_tensors = Variable(torch.Tensor(X_train).to(device))
X_test_tensors = Variable(torch.Tensor(X_test).to(device))

y_train_tensors = Variable(torch.Tensor(y_train).to(device))
y_test_tensors = Variable(torch.Tensor(y_test).to(device)) 

X_train_tensors_final = torch.reshape(X_train_tensors,   (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))
X_test_tensors_final = torch.reshape(X_test_tensors,  (X_test_tensors.shape[0], 1, X_test_tensors.shape[1])) 

lstm1 = athenaLTSM(num_classes, input_size, hidden_size, num_layers, seq_length) #our lstm class
lstm1.to(device)

criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate) 

print('Training with: ', device)
for epoch in range(num_epochs):
  outputs = lstm1.forward(X_train_tensors_final) #forward pass
  optimizer.zero_grad() #caluclate the gradient, manually setting to 0
 
  # obtain the loss function
  loss = criterion(outputs, y_train_tensors)
 
  loss.backward() #calculates the loss of the loss function
 
  optimizer.step() #improve from loss, i.e backprop
  if epoch % 100 == 0:
    print("Epoch: %d, loss: %1.5f" % (epoch, loss.item())) 

print("finished training")

df_X_ss = ss.transform(df.iloc[:, 3:]) #old transformers
df_y_mm = mm.transform(df.iloc[:, 2:3]) #old transformers

df_X_ss = Variable(torch.Tensor(df_X_ss).to(device)) #converting to Tensors
df_y_mm = Variable(torch.Tensor(df_y_mm).to(device))
#reshaping the dataset
df_X_ss = torch.reshape(df_X_ss, (df_X_ss.shape[0], 1, df_X_ss.shape[1]))

train_predict = lstm1(df_X_ss)#forward pass
data_predict = train_predict.data.cpu().numpy() #numpy conversion
dataY_plot = df_y_mm.data.cpu().numpy()

data_predict = mm.inverse_transform(data_predict) #reverse transformation
dataY_plot = mm.inverse_transform(dataY_plot)
plt.figure(figsize=(10,6)) #plotting
plt.axvline(x=split, c='r', linestyle='--') #size of the training set

plt.plot(dataY_plot, label='Actuall Data') #actual plot
plt.plot(data_predict, label='Predicted Data') #predicted plot
plt.title('Time-Series Prediction')

torch.save(lstm1, './models/savedModels/mk1.pth')

plt.legend()
plt.show() 