import sys
import count1
import random
import math
fname = sys.argv[1]
n = int(sys.argv[2])
fn = sys.argv[3]

N,dc= count1.run()
#print N
#print n

index2=list(xrange(23))
index=list(xrange(23))
p=list(xrange(23))

print dc.keys()[1]
for i in range(len(index)):
	index[i]= dc[dc.keys()[i]][1]
#	print index[i]
	p[i] = math.ceil((float(n)*float(dc[dc.keys()[i]][0]))/float(N))

ind = []
for i in range(len(index)):
	print index[i], p[i]
	index2[i] = random.sample(index[i],int(p[i]))
	ind = index2[i]+ind

#ind = random.sample(xrange(0,N), n)
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
