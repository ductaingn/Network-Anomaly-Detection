import pandas as pd
import pickle as pk
import matplotlib.pyplot as plt
data = pd.read_csv('./Data/Train.txt')
# print(data['last_flag'].unique())
print(data.head())