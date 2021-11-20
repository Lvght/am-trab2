import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import csv

file = open('datasets/student_data_por.csv')
csvreader = csv.reader(file)
header = next(csvreader)
data = []

for row in csvreader:
  data.append(row)


enc = OneHotEncoder(handle_unknown='ignore');
X = enc.fit_transform(data).toarray()

scaler = StandardScaler(with_mean=False).fit(X)
X = scaler.transform(X)

print(X)
print('Len is ', len(X))
print('Instance size is ', X[0].size)

# Implemente aqui o seu algoritmo.