import csv
import numpy as np
from random import randrange
from copy import deepcopy
import cvxopt
import cvxopt.solvers
from matplotlib import pyplot as plt
import sys
import math
from texttable import Texttable
import distinct
from sklearn.svm import SVC
import ffr
import timeit 

def load_inputfile(filename, features, l):
	X = []
	y = []
	c1 = []
	c2 = []
	data = distinct.preprocess(filename, 'svm')
#	l = []
#	if features < 41:
#		l = ffr.feature_reduction(data)
	for row in data:
		X.append(row[:-1])
		y.append(row[-1])	
		if row[-1] == 1:
	       		c1.append(row[:-1])
		else:
			c2.append(row[:-1])
		if features < 41:
			X[-1] = [X[-1][i] for i in l]
	c1 = np.array(c1,dtype='float')
	c2 = np.array(c2,dtype='float')
#	y1 = np.ones(len(x1))
#	y2 = np.ones(len(x2)) * -1
#	X = np.array(X,dtype='float')
#	y = np.array(y,dtype='float')
	return X,y,c1,c2

def polynomial_kernel(x, y, p):
	return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma):
	return np.exp(-np.linalg.norm(x - y)**2 / (2 * (sigma ** 2)))

#*********************Trying something*************************/
class Non_Linear_SVM(object):
	def __init__(self,kernel,kernel_param,C):
		self.kernel = kernel
		self.kernel_param = kernel_param
		self.C = C

	def train(self, X, y):
		n_samples, n_features = X.shape
		self.K = np.zeros((n_samples, n_samples))
		for i in range(n_samples):
			for j in range(n_samples):
				self.K[i, j] = self.kernel(X[i], X[j], kernel_param)
		P = cvxopt.matrix(np.outer(y, y) * self.K)
		q = cvxopt.matrix(np.ones(n_samples) * -1)
		A = cvxopt.matrix(y, (1,n_samples))
		b = cvxopt.matrix(0.0)

		tmp1 = np.diag(np.ones(n_samples) * -1)
		tmp2 = np.identity(n_samples)
		G = cvxopt.matrix(np.vstack((tmp1,tmp2)))
		tmp1 = np.zeros(n_samples)
		tmp2 = np.ones(n_samples) * self.C
		h = cvxopt.matrix(np.hstack((tmp1,tmp2)))
		
		cvxopt.solvers.options['show_progress'] = False
		solution = cvxopt.solvers.qp(P, q, G, h, A, b)

		a = np.ravel(solution['x'])

		self.sv = a > 1e-5
		self.sv_ind = np.arange(len(a))[self.sv]
		self.a = a[self.sv]
		self.sv_X = X[self.sv]
		self.sv_y = y[self.sv]
		
	def classify(self, X):
		self.bias = 0
		for i in range(len(self.a)):
			self.bias += (self.sv_y[i] - np.sum(self.a * self.sv_y * self.K[self.sv_ind[i], self.sv]))
		self.bias /= len(self.a)
		y_predict = np.zeros(len(X))
		for i in range(len(X)):
			for a, sv_y, sv_X in zip(self.a, self.sv_y, self.sv_X):
				y_predict[i] += (a * sv_y * self.kernel(X[i], sv_X, self.kernel_param))
		result = np.sign(y_predict + self.bias)
		return result

#*********************not anymore******************************/


def evaluate_accuracy(y_predict, y_testing, no_of_testing_samples):
	correct = np.sum(y_predict == np.array(y_testing,dtype='float'))
	return (float(correct)/float(no_of_testing_samples))*100

if __name__ == "__main__":
	choice = 'y'
 
#	kernel = gaussian_kernel
	kernel = 'rbf'
	C = 10000
	kernel_param = 0.000001
	
	while choice != 'n':
		features = input("Enter number of features: ")
		filename = raw_input("Enter training dataset filename: ")
		data = distinct.preprocess(filename, 'svm')
		l = []
		if features < 41:
			l = ffr.feature_reduction(data)
		l = l[:features]
#		print l
		X_training,y_training,c1_training,c2_training = load_inputfile(filename, features, l)
		filename = raw_input("Enter test dataset filename: ")
		X_testing, y_testing,c1_testing, c2_testing = load_inputfile(filename, features, l)
		no_of_testing_samples = len(X_testing)

#		obj = Non_Linear_SVM(kernel,kernel_param,C)
#		obj.train(np.array(X_training,dtype='float'),np.array(y_training,dtype='float'))
#		y_predict = obj.classify(np.array(X_testing,dtype='float'))
		start_time = timeit.default_timer()
		
		obj = SVC(C=C,kernel=kernel,gamma=kernel_param)
		obj.fit(np.array(X_training,dtype='float'), np.array(y_training,dtype='float'))
	
		elapsed1 = timeit.default_timer() - start_time

		start_time = timeit.default_timer()

		y_predict = obj.predict(np.array(X_testing,dtype='float'))
		
		elapsed2 = timeit.default_timer() - start_time

		accuracy = evaluate_accuracy(y_predict, y_testing, no_of_testing_samples)

		result = []
		result.append([C] + [kernel_param] + [accuracy] + [elapsed1] + [elapsed2])
		print "Results:"
		result.insert(0,['C','sigma','Accuracy', 'Training time', 'Testing time'])
		t = Texttable()
		t.set_precision(15)
		t.add_rows(result)
		print t.draw()

		choice = raw_input("Continue? (y/n) ")
