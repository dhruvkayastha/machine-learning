# Dhruv Kayastha
# 16CS30041
# Assignment 3 - AdaBoost

from csv import reader
import numpy as np
import math
import random
from bisect import bisect_right
from time import time

random.seed(time())
class DT(object):
	def __init__(self, is_terminal=False):
		self.is_root = False
		self.num_children = 0
		self.left = None
		self.mid_left = None
		self.mid_right = None
		self.right = None
		self.parent_attribute = None
		self.parent_attribute_val = None
		self.attribute = None
		self.is_terminal = is_terminal
		self.terminal_class = None

def split_current_node(attribute, dataset, attribute_values):
	if attribute != 0:
		left_child = np.empty((0,4), dtype='<U4')
		right_child = np.empty((0,4), dtype='<U4')
		for row in dataset:
			if row[attribute] == attribute_values[attribute][0]:
				left_child = np.concatenate((left_child, np.array([row])))
			elif row[attribute] == attribute_values[attribute][1]:
				right_child = np.concatenate((right_child, np.array([row])))
		return left_child, right_child
	else:
		left_child = np.empty((0,4), dtype='<U4')
		middle_left_child = np.empty((0,4), dtype='<U4')
		middle_right_child = np.empty((0,4), dtype='<U4')
		right_child = np.empty((0,4), dtype='<U4')
		for row in dataset:
			if row[attribute] == attribute_values[attribute][0]:
				left_child = np.concatenate((left_child, np.array([row])))
			elif row[attribute] == attribute_values[attribute][1]:
				middle_left_child = np.concatenate((middle_left_child, np.array([row])))
			elif row[attribute] == attribute_values[attribute][2]:
				middle_right_child = np.concatenate((middle_right_child, np.array([row])))
			elif row[attribute] == attribute_values[attribute][3]:	
				right_child = np.concatenate((right_child, np.array([row])))
		return left_child, middle_left_child, middle_right_child, right_child

def calc_info_gain(parent_group, groups, classes):
	total_instances = float(sum([len(group) for group in groups]))
	entropy = 0.0
	for group in groups:
		score = 0.0
		size = float(len(group))
		if size == 0:
			continue
		for class_val in classes:
			p = [row[-1] for row in group].count(class_val)/size
			if p != 0:
				score += -(p*math.log(p,2))
		entropy += score*(size/total_instances)
	parent_entropy = 0.0
	size = float(len(parent_group))
	for class_val in classes:
		p = [row[-1] for row in parent_group].count(class_val)/size
		parent_entropy += -(p*math.log(p,2))
	return parent_entropy - entropy

def get_best_split(dataset, classes, attribute_values, attributes_left):
	max_info_gain = -100000
	best_attr = -1
	best_split = []
	for attr in attributes_left:
		groups = split_current_node(attr, dataset, attribute_values)
		info_gain = calc_info_gain(dataset, groups, classes)
		if info_gain > max_info_gain:
			max_info_gain = info_gain
			best_attr = attr
			best_split = groups	    	
	return best_attr, best_split

def fit(node, group, classes, attribute_values, attributes_unused_t):
	count = []
	cur_count = 0
	attributes_unused = attributes_unused_t.copy()
	for class_val in classes:
		cur_count = 0
		for row in group:
			if row[-1] == class_val:
				cur_count += 1
		count.append(cur_count)
	if count[0] == len(group) or count[1] == len(group) or node.is_terminal is True:
		node.is_terminal = True
		if count[0] > count[1]:
			node.terminal_class = 'yes'
		else:
			node.terminal_class = 'no'
	else:
		# node.is_terminal = False
		best_attr, best_split_groups = get_best_split(group, classes, attribute_values, attributes_unused)
		# if node.is_root == True:
		# 	print("\nInformation Gain @ root node =", calc_info_gain(group, best_split_groups, classes))
		if best_attr == -1:
			node.is_terminal = True
			if count[0] > count[1]:	
				node.terminal_class = 'yes'
			else:
				node.terminal_class = 'no'
			return

		attributes_unused.remove(best_attr)
		node.attribute = best_attr
		if len(best_split_groups) == 2:			
			node.left = DT(is_terminal=True)
			node.left.parent_attribute = best_attr
			node.left.parent_attribute_val = 0
			fit(node.left, best_split_groups[0], classes, attribute_values, attributes_unused)
			
			node.right = DT(is_terminal=True)
			node.right.parent_attribute = best_attr
			node.right.parent_attribute_val = 1
			fit(node.right, best_split_groups[1], classes, attribute_values, attributes_unused)
			node.num_children = 2

		else:		#split for pclass attribute
			node.left = DT(is_terminal=True)
			node.left.parent_attribute = best_attr
			node.left.parent_attribute_val = 0
			fit(node.left, best_split_groups[0], classes, attribute_values, attributes_unused)

			node.mid_left = DT(is_terminal=True)
			node.mid_left.parent_attribute = best_attr
			node.mid_left.parent_attribute_val = 1
			fit(node.mid_left, best_split_groups[1], classes, attribute_values, attributes_unused)

			node.mid_right = DT(is_terminal=True)
			node.mid_right.parent_attribute = best_attr
			node.mid_right.parent_attribute_val = 2
			fit(node.mid_right, best_split_groups[2], classes, attribute_values, attributes_unused)
			
			node.right = DT(is_terminal=True)
			node.right.parent_attribute = best_attr
			node.right.parent_attribute_val = 3
			fit(node.right, best_split_groups[3], classes, attribute_values, attributes_unused)
			node.num_children = 4

def print_tree(node, attributes_list, attribute_values, level):
	if node is None:
		return
	if node.is_root == True:
		if node.num_children == 2:
			level = 1
			print_tree(node.left, attributes_list, attribute_values, level)
			print_tree(node.right, attributes_list, attribute_values, level)
		else:
			level = 1
			print_tree(node.left, attributes_list, attribute_values, level)
			print_tree(node.mid_left, attributes_list, attribute_values, level)
			print_tree(node.mid_right, attributes_list, attribute_values, level)
			print_tree(node.right, attributes_list, attribute_values, level)
	else:
		if node.is_terminal	== True:
			for i in range(level-2):
				print('\t', end=" ")
			if(level > 1):
				print("|", attributes_list[node.parent_attribute], "=", attribute_values[node.parent_attribute][node.parent_attribute_val], ":", node.terminal_class)
			else:
				print(attributes_list[node.parent_attribute], "=", attribute_values[node.parent_attribute][node.parent_attribute_val], ":", node.terminal_class)
		else:
			for i in range(level-2):
				print('\t', end=" ")
			if(level > 1):
				print("|", attributes_list[node.parent_attribute], "=", attribute_values[node.parent_attribute][node.parent_attribute_val])
			else:
				print(attributes_list[node.parent_attribute], "=", attribute_values[node.parent_attribute][node.parent_attribute_val])
			if node.num_children == 2:
				level += 1
				print_tree(node.left, attributes_list, attribute_values, level)
				print_tree(node.right, attributes_list, attribute_values, level)
			else:
				level += 1
				print_tree(node.left, attributes_list, attribute_values, level)
				print_tree(node.mid_left, attributes_list, attribute_values, level)
				print_tree(node.mid_right, attributes_list, attribute_values, level)
				print_tree(node.right, attributes_list, attribute_values, level)

def predict(node, data, attribute_values):
	if node.is_terminal == True:
		return node.terminal_class
	else:
		if node.num_children == 4:
			if data[node.attribute] == attribute_values[node.attribute][0]:
				return predict(node.left, data, attribute_values)
			elif data[node.attribute] == attribute_values[node.attribute][1]:
				return predict(node.mid_left, data, attribute_values)
			elif data[node.attribute] == attribute_values[node.attribute][2]:
				return predict(node.mid_right, data, attribute_values)
			else:
				return predict(node.right, data, attribute_values)
		else:
			if data[node.attribute] == attribute_values[node.attribute][0]:
				return predict(node.left, data, attribute_values)
			else:
				return predict(node.right, data, attribute_values)

train_filename = 'data3_19.csv'
file = open(train_filename, "r")
lines = reader(file)
dataset = list(lines)

del dataset[0]
dataset = np.array(dataset)



attributes_list = ['pclass', 'age', 'gender']
attribute_values = [['1st', '2nd', '3rd', 'crew'], ['adult', 'child'], ['male', 'female']]
classes = ['yes', 'no']
attributes_left = [0,1,2]


tree1 = DT()
tree1.is_root = True
fit(tree1, dataset, classes, attribute_values, attributes_left)

# print("\nDecision Tree: \n")
# print_tree(tree1, attributes_list, attribute_values, 0)

count = 0
n = len(dataset)
weights = np.full((n), 1.0/n)
total_error = 0.0

correctly_classified_idx = {}

for i, data in enumerate(dataset):
	predicted_label = predict(tree1, data, attribute_values)
	true_label = data[3]
	if predicted_label == true_label:
		count += 1
		correctly_classified_idx[i] = 1
	else:
		total_error += weights[i]

sig = 0.5*np.log((1 - total_error)/total_error)

for i in range(n):
	if i in correctly_classified_idx:
		weights[i] *= np.exp(-sig)
	else:
		weights[i] *= np.exp(sig)
weights /= sum(weights)
prob_interval = np.cumsum(weights)

dataset2 = np.copy(dataset)

for i in range(n):
	rand = random.uniform(0, 1)

	idx = bisect_right(prob_interval, rand)
	dataset2[i] = dataset[idx]
	# if i == 0:
	# 	for j in range(n):
	# 		if prob_interval[j] > rand:
	# 			print("j",j)
	# 			break
	# 	print("idx:", idx)
	# 	print("random", rand)
	# 	print(prob_interval)

########################################################

tree2 = DT()
tree2.is_root = True
fit(tree2, dataset2, classes, attribute_values, attributes_left)

total_error = 0.0

correctly_classified_idx = {}

for i, data in enumerate(dataset2):
	predicted_label = predict(tree2, data, attribute_values)
	true_label = data[3]
	if predicted_label == true_label:
		count += 1
		correctly_classified_idx[i] = 1
	else:
		total_error += weights[i]

sig2 = 0.5*np.log((1 - total_error)/total_error)

for i in range(n):
	if i in correctly_classified_idx:
		weights[i] *= np.exp(-sig2)
	else:
		weights[i] *= np.exp(sig2)
weights /= sum(weights)
correctly_classified_idx = {}

prob_interval = np.cumsum(weights)

dataset3 = np.copy(dataset2)

for i in range(n):
	rand = random.uniform(0, 1)
	idx = bisect_right(prob_interval, rand)
	dataset3[i] = dataset2[idx]

#########################################################################3


tree3 = DT()
tree3.is_root = True
fit(tree3, dataset3, classes, attribute_values, attributes_left)

total_error = 0.0

correctly_classified_idx = {}

for i, data in enumerate(dataset3):
	predicted_label = predict(tree3, data, attribute_values)
	true_label = data[3]
	if predicted_label == true_label:
		count += 1
		correctly_classified_idx[i] = 1
	else:
		total_error += weights[i]

sig3 = 0.5*np.log((1 - total_error)/total_error)

# print_tree(tree1, attributes_list, attribute_values, 0)
# print_tree(tree2, attributes_list, attribute_values, 0)
# print_tree(tree3, attributes_list, attribute_values, 0)


count = 0
for data in dataset:
	predicted_label1 = predict(tree1, data, attribute_values)
	predicted_label2 = predict(tree2, data, attribute_values)
	predicted_label3 = predict(tree3, data, attribute_values)

	# print(predicted_label1, predicted_label2, predicted_label3)

	predicted_label1 = -1.0 if predicted_label1 == 'no' else 1.0
	predicted_label2 = -1.0 if predicted_label2 == 'no' else 1.0
	predicted_label3 = -1.0 if predicted_label3 == 'no' else 1.0

	# if predicted_label2 == 0:
	# 	predicted_label2 = -1.0
	# if predicted_label3 == 0:
	# 	predicted_label3 = -1.0

	boosted_pred = sig*predicted_label1 + sig2*predicted_label2 + sig3*predicted_label3

	if boosted_pred > 0.0:
		boosted_pred = 'yes'
	else:
		boosted_pred = 'no'

	if boosted_pred == data[3]:
		count += 1

accuracy = count*100.0/(len(dataset))
print("AdaBoost Training Accuracy =", accuracy)

train_filename = 'test3_19.csv'
file = open(train_filename, "r")
lines = reader(file)
dataset = list(lines)

# del dataset[0]
dataset = np.array(dataset)

count = 0
for data in dataset:
	predicted_label1 = predict(tree1, data, attribute_values)
	predicted_label2 = predict(tree2, data, attribute_values)
	predicted_label3 = predict(tree3, data, attribute_values)

	# print(predicted_label1, predicted_label2, predicted_label3)

	predicted_label1 = -1.0 if predicted_label1 == 'no' else 1.0
	predicted_label2 = -1.0 if predicted_label2 == 'no' else 1.0
	predicted_label3 = -1.0 if predicted_label3 == 'no' else 1.0

	# if predicted_label2 == 0:
	# 	predicted_label2 = -1.0
	# if predicted_label3 == 0:
	# 	predicted_label3 = -1.0

	boosted_pred = sig*predicted_label1 + sig2*predicted_label2 + sig3*predicted_label3

	if boosted_pred > 0.0:
		boosted_pred = 'yes'
	else:
		boosted_pred = 'no'

	if boosted_pred == data[3]:
		count += 1

accuracy = count*100.0/(len(dataset))
print("AdaBoost Training Accuracy =", accuracy)
