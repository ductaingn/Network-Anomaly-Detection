import pandas as pd
import pickle as pk
import matplotlib.pyplot as plt

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle

def save(data,file_name):
    name = './Data/'+file_name+'.pickle'
    file = open(name,'wb')
    pickle.dump(data,file)
    print(f"Saved {file_name}!")
            
def load(file_name):
    name = './Data/'+file_name+'.pickle'
    file = open(name,'rb')
    res = pickle.load(file)
    return res

# def increase_epoch_count(num_of_epoch):
#     with open('Epoch Counter.txt','r') as file:
#         count = int(file.readlines()[1])

#     with open('Epoch Counter.txt','r') as file:
#         change = file.read().replace(str(count), str(count + num_of_epoch))

#     with open('Epoch Counter.txt', 'w') as file:
#         file.write(change)

#     print(f'Total number of epoch trained: {count+num_of_epoch}')

    