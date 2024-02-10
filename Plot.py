import matplotlib.pyplot as plt
import pickle
import torch
import numpy as np

def plot_loss():
    loss = torch.load('model.pt')['losses']

    p = []
    for i in range(len(loss)):
        p.append(np.mean(loss[0:i]))
    plt.plot(loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Mean Average Loss')
    plt.show()