import sys
fname = sys.argv[1]

def run():
	with open(fname) as f:
		for i,l in enumerate(f):
			pass
#	print i + 1
	return i + 1
