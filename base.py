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
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt

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

# Agrupamento Hierárquico
agglomerative = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
model = agglomerative.fit(X)

counts = np.zeros(model.children_.shape[0])
n_samples = len(model.labels_)
for i, merge in enumerate(model.children_):
    current_count = 0
    for child_idx in merge:
        if child_idx < n_samples:
            current_count += 1  # leaf node
        else:
            current_count += counts[child_idx - n_samples]
    counts[i] = current_count

linkage_matrix = np.column_stack(
    [model.children_, model.distances_, counts]
).astype(float)

dendrogram(linkage_matrix, truncate_mode="level", p=3)


