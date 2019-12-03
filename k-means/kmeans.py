# Dhruv Kayastha
# 16CS30041
# Assignment 4: K-Means

import csv
import numpy as np
import math

np.random.seed(30041)	# roll number

def mode(x):
	return max(set(x), key = x.count)

def load_csv(filename):
	lines = csv.reader(open(filename, "r"))
	dataset = list(lines)
	return dataset[:-1]

def split_XY(dataset):
	y = [data[-1] for data in dataset]
	x = [data[:-1] for data in dataset]
	x = [[float(d) for d in data] for data in x]
	return x, y

train_dataset = load_csv('data4_19.csv')
x_train, y_train = split_XY(train_dataset)
x_train = np.array(x_train)

n = len(y_train)
# print(n)
centers = np.random.randint(0, high=150, size=(3))
centers = [x_train[i] for i in centers]

for iteration in range(0, 10):
	clusters = [[],[],[]]

	for i, x in enumerate(x_train):
		d = np.zeros((3))
		d[0] = np.linalg.norm(x - centers[0])
		d[1] = np.linalg.norm(x - centers[1])
		d[2] = np.linalg.norm(x - centers[2])

		nearest_cluster = np.argmin(d)

		clusters[nearest_cluster].append(i)

	centers[0] = np.mean(x_train[clusters[0]], axis=0)
	centers[1] = np.mean(x_train[clusters[1]], axis=0)
	centers[2] = np.mean(x_train[clusters[2]], axis=0)

	# print(len(centers))
	# print(centers)

print("Final Cluster Means:")
print("Cluster 1:", centers[0])
print("Cluster 2:", centers[1])
print("Cluster 3:", centers[2])


cluster_labels = []
ground_truth = [[],[],[]]

for c in clusters:
	cluster_labels.append(mode([y_train[i] for i in c]))

for j in range(n):
	for i, label in enumerate(cluster_labels):
		if y_train[j] == label:
			ground_truth[i].append(j)

for i, label in enumerate(cluster_labels):
	print("\nClass Label:",label)
	union_size = len(set(ground_truth[i]).union(set(clusters[i])))
	inter_size = len(set(ground_truth[i]).intersection(set(clusters[i])))

	jDist = 1.0 - inter_size/union_size

	print("Jaccard Distance:", jDist)






