from sklearn.neural_network import MLPClassifier
import numpy as np
import sys
import math
from texttable import Texttable
import distinct
import ffr
import timeit 
#from nn1 import *

def load_inputfile(filename, features, l):
	X = []
	y = []
	data = distinct.preprocess(filename, 'nn')
	for row in data:
		X.append(row[:-1])
		y.append(row[-1])	
		if features < 41:
			X[-1] = [X[-1][i] for i in l]
	return X,y
	
def evaluate_accuracy(y_predict, y_testing, no_of_testing_samples):
	correct = np.sum(y_predict == np.array(y_testing,dtype='float'))
	return (float(correct)/float(no_of_testing_samples))*100

if __name__ == "__main__":
	choice = 'y'
 
	C = 10000
	kernel_param = 0.000001
	
	while choice != 'n':
		features = input("Enter number of features: ")
		filename = raw_input("Enter training dataset filename: ")
		data = distinct.preprocess(filename, 'nn')
		l = []
		if features < 41:
			l = ffr.feature_reduction(data)
		l = l[:features]
#		print l
		X_training,y_training = load_inputfile(filename, features, l)
		filename = raw_input("Enter test dataset filename: ")
		X_testing, y_testing = load_inputfile(filename, features, l)
		hl1 = input("Enter number of nodes for hidden layer 1: ")
		hl2 = input("Enter number of nodes for hidden layer 2: ")
		epochs = input("Enter number of epochs: ")

		no_of_testing_samples = len(X_testing)

		arch = [features, hl1, hl2, 5]
#		shape = [features, 50, 5]

		start_time = timeit.default_timer()

		obj = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(hl1, hl2), random_state=1, max_iter = epochs)
		obj.fit(np.array(X_training,dtype='float'), np.array(y_training,dtype='float'))

#		weights = generate_network(shape)
#		weights = train_network_main(X_training, y_training, 0.01, shape, weights)

		elapsed1 = timeit.default_timer() - start_time

		start_time = timeit.default_timer()

#		for i in range(10):
#		    y_predict = run_network(X_testing, shape, weights)

		y_predict = obj.predict(np.array(X_testing,dtype='float'))
		
		elapsed2 = timeit.default_timer() - start_time

		accuracy = evaluate_accuracy(y_predict, y_testing, no_of_testing_samples)

		result = []

		result.append([arch] + [accuracy] + [elapsed1] + [elapsed2])
		print "Results:"
		result.insert(0,['Architecture','Accuracy', 'Training time', 'Testing time'])
		t = Texttable()
		t.set_precision(15)
		t.add_rows(result)
		print t.draw()

		choice = raw_input("Continue? (y/n) ")
