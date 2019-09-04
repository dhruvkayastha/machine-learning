# Dhruv Kayastha
# 16CS30041
# Assignment 2 - Naive Bayes Classifier

import csv
import numpy as np
import math

def load_csv(filename):
	lines = csv.reader(open(filename, "rt"))
	dataset = list(lines)
	del dataset[0]
	for i in range(len(dataset)):
		dataset[i] = list(csv.reader(dataset[i]))
	dataset = np.array(dataset)
	dataset = np.squeeze(dataset, axis=1)
	dataset = dataset.astype(int)
	return dataset

def split_XY(dataset):
	y = dataset[:, 0]
	x = dataset[:, 1:]
	return x, y

class NaiveBayesClassifier:
	def __init__(self):
		self.dataset = {}
		self.summaries = None

	def splitByClass(self, dataset):
		split = {0:[], 1:[]}
		for data in dataset:
			split[data[0]].append(data[1:])
		self.dataset[0] = np.array(split[0])
		self.dataset[1] = np.array(split[1])

	def summarize(self, dataset):
		summary = np.zeros((6, 6)) 
		for row in dataset:
			for i in range(len(row)):
				summary[i][row[i]]+=1
		return summary

	def train(self, dataset):
		self.summaries = {}
		self.splitByClass(dataset)
		self.summaries[0] = self.summarize(self.dataset[0])
		self.summaries[1] = self.summarize(self.dataset[1])

	def calculateProbability(self, X_vector, classVal):
		P = 1.
		Nc = len(self.dataset[classVal])
		M = len(self.dataset[1 - classVal])
		
		for i in range(len(X_vector)):
			Nic = self.summaries[classVal][i][X_vector[i]]
			p = (Nic + 1.)/(Nc + len(X_vector))	#Laplacian smoothing
			P *= p
		
		return P*(1.*Nc/(Nc + M))

	def predict(self, X_vector):	#single instance of data point
		P0 = self.calculateProbability(X_vector, 0)
		P1 = self.calculateProbability(X_vector, 1)
		if P0 > P1:
			return 0
		return 1

	def accuracy(self, x, y):
		n = len(x)
		count = 0
		for i in range(len(x)):
			y_pred = self.predict(x[i])
			if y_pred == y[i]:
				count+=1
		return 100.0*count/n

train_dataset = load_csv('data2_19.csv')
x_train, y_train = split_XY(train_dataset)
test_dataset = load_csv('test2_19.csv')
x_test, y_test = split_XY(test_dataset)

model = NaiveBayesClassifier()
model.train(train_dataset)

training_acc = model.accuracy(x_train, y_train)
print("Training accuracy:", training_acc)

test_acc = model.accuracy(x_test, y_test)
print("Test accuracy:", test_acc)
