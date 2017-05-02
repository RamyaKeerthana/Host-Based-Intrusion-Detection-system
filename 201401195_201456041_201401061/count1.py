import sys
fname = sys.argv[1]

#dic = {'normal':0, 'back':0, 'buffer_overflow':0, 'ftp_write':0, 'guess_passwd':0, 'imap':0, 'ipsweep':0, 'land':0, 'loadmodule':0, 'multihop':0, 'neptune':0, 'nmap':0, 'perl':0, 'phf':0, 'pod':0, 'portsweep':0, 'rootkit':0, 'satan':0, 'smurf':0, 'spy':0, 'teardrop':0, 'warezclient':0, 'warezmaster':0}

def run():

	dic = {'normal':[0,[]], 'back':[0,[]], 'buffer_overflow':[0,[]], 'ftp_write':[0,[]], 'guess_passwd':[0,[]], 'imap':[0,[]], 'ipsweep':[0,[]], 'land':[0,[]], 'loadmodule':[0,[]], 'multihop':[0,[]], 'neptune':[0,[]], 'nmap':[0,[]], 'perl':[0,[]], 'phf':[0,[]], 'pod':[0,[]], 'portsweep':[0,[]], 'rootkit':[0,[]], 'satan':[0,[]], 'smurf':[0,[]], 'spy':[0,[]], 'teardrop':[0,[]], 'warezclient':[0,[]], 'warezmaster':[0,[]]}
	with open(fname) as f:
		for i,l in enumerate(f):
			l = l.split(',')
			x = l[-1]
			x = x.rstrip('.\n')
			dic[x][0] += 1
			dic[x][1].append(i)
	
#	print i + 1
#	print dic
	return i + 1,dic
