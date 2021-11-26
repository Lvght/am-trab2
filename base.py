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
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt
import csv

file = open('tripadvisor_review.csv')
csvreader = csv.reader(file)
header = next(csvreader)
data = []

for row in csvreader:
  data.append(np.array(row[1:], dtype=float))

# Implemente aqui o seu algoritmo.

# Agrupamento Hierárquico
best_silhouette_score = -1;
best_model = None;
best_n = None;

silhouette_array = []

for i in range(2, 20, 2):
  agglomerative = AgglomerativeClustering(distance_threshold=None, n_clusters=i, compute_distances=True)
  model = agglomerative.fit(data)

  silhouette = silhouette_score(data, model.labels_)
  silhouette_array.append(silhouette)
  if(silhouette > best_silhouette_score):
    best_silhouette_score = silhouette
    best_model = model
    best_n = i

model = best_model
counts = np.zeros(model.children_.shape[0])
n_samples = len(model.labels_)

for i, merge in enumerate(model.children_):
    current_count = 0
    for child_idx in merge:
        if child_idx < n_samples:
            current_count += 1  
        else:
            current_count += counts[child_idx - n_samples]
    counts[i] = current_count

linkage_matrix = np.column_stack(
    [model.children_, model.distances_, counts]
).astype(float)

plt.plot(range(2, 20, 2), silhouette_array)

plt.xlabel('Clusters')
plt.ylabel('Silhouette score')

plt.show()

print("A melhor silhueta é", best_silhouette_score)
print("O melhor número de clusters é", best_n)
dendrogram(linkage_matrix, truncate_mode="level", p=best_n)
plt.show()