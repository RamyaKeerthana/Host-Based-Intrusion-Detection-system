import sys

#fn = sys.argv[2]

def preprocess(fname, classifier):
	distinct_f = [[]]*3
	dict = {'back':1, 'land':1, 'neptune' :1 , 'pod':1, 'smurf':1, 'teardrop':1, 'buffer_overflow':2 , 'loadmodule':2, 'perl':2, 'rootkit':2, 'ftp_write':3, 'guess_passwd':3, 'imap':3, 'multihop':3, 'spy':3, 'warezclient':3, 'warezmaster':3, 'phf':3, 'ipsweep':4, 'nmap':4, 'portsweep':4, 'satan':4, 'normal':0}
	data = []
	with open(fname) as f:
		for l in f.readlines():
			l = l.rstrip('.\n')
			l = l.split(',')
			for i in range(len(l)):
				if i == len(l) - 1:
					if classifier == 'svm':
			       			if l[-1] == 'normal':
							l[-1] = 1
						else:
							l[-1] = -1
					else:
						l[-1] = dict[l[-1]]
				elif i == 1 or i == 2 or i == 3:
					if l[i] in distinct_f[i-1]:
						l[i] = distinct_f[i-1].index(l[i])
					else:
						distinct_f[i-1].append(l[i])
						l[i] = distinct_f[i-1].index(l[i])
			data.append(l)
	#print data[104]
	return data
