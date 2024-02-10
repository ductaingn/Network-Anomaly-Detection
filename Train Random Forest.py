import pandas as pd
import pickle as pk
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import IO

X_train = IO.load('X_train')
X_test = IO.load('X_test')
Y_train = IO.load('Y_train')
Y_test = IO.load('Y_test')

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train,Y_train)

y_predict = rf.predict(X_test)
score = rf.score(X_test,Y_test)
print(f'Accuracy: {score*100}%')
