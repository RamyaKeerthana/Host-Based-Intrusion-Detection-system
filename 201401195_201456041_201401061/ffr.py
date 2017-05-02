import distinct
import sys
import math

def feature_reduction(data):
	d = 41
	c = 2
	c1 = 0
	c2 = 0
	V = [[0.0]*c]*d
	m = [0.0]*d
#filename = sys.argv[1]
#data = distinct.preprocess(filename)
	for row in data:
		for i in range(len(row)-1):
			if row[-1] == 1:			
				V[i][0] += float(row[i])
				c1 += 1
			else:
				V[i][1] += float(row[i])
				c2 += 1

	for i in range(len(V)):
		V[i][0] = float(V[i][0])/float(c1)
		V[i][1] = float(V[i][1])/float(c2)
		m[i] += float(V[i][0] + V[i][1])/float(c)
	
	var = [0.0]*d
	for i in range(len(var)):
		var[i] = float(math.pow((V[i][0] - m[i]),2) + math.pow((V[i][1] - m[i]),2))/float(c)

	l = sorted(range(len(var)), key=lambda k: var[k])
	return l

#filename = sys.argv[1]
#data = distinct.preprocess(filename)
#l = feature_reduction(data)
#print l
