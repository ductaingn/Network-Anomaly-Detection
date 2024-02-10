import torch
import torch.nn as nn
import torch.nn.functional as func
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import IO
from Model import Model

X_train = IO.load('X_train')
X_test = IO.load('X_test')
Y_train = IO.load('Y_train')
Y_test = IO.load('Y_test')

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
Y_train = torch.LongTensor(Y_train)
Y_test = torch.LongTensor(Y_test)

torch.manual_seed(39)
model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

try:
    checkpoint = torch.load('model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    losses = checkpoint['losses']
    epoch_counter = checkpoint['epoch_counter']
    for group in optimizer.param_groups:
        group['lr'] *= (torch.e**-1)
    print("Model loaded succesfully!")
except FileNotFoundError:
    epoch = 1000
    epoch_counter = 0
    losses = []

for i in range(epoch):
    Y_pred = model.forward(X_train)
    loss = criterion(Y_pred, Y_train)
    losses.append(loss.detach().numpy())
    if i%10==0:
        print(f"Epoch: {i+epoch_counter} ,Loss: {losses[i+epoch_counter]}")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.save({
    'epoch': epoch,
    'epoch_counter': epoch_counter + epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'losses': losses
},'model.pt')

count = 0
with torch.no_grad():
    for i,data in enumerate(X_test):
        Y_eval = model.forward(data)

        if Y_eval.argmax().item() == Y_test[i]:
            count+=1

print(f'Accuracy: {count/len(Y_test)*100}%')