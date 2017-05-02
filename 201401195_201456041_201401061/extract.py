import sys
import count
import random
fname = sys.argv[1]
n = int(sys.argv[2])
fn = sys.argv[3]

N = int(count.run())
print N
print n


ind = random.sample(xrange(0,N), n)
#print ind
lst = []
with open(fname) as f:
	for i,l in enumerate(f):
		if i in ind:
#			print i
#			l = l.rstrip('\n')
			lst.append(l)
		print i
print len(lst)		
#print lst

with open(fn,"a") as f:
	for i in range(len(lst)):
		f.write(lst[i])
