# Loads and calls a c library function
# Run using: python arrayuse.py

import numpy as np
from numpy import ctypeslib
from ctypes import *
import math
import multiprocessing
import sys

#kVal = int(sys.argv[1])

# Loads the c library
arraytest = cdll.LoadLibrary('./doWork.so.1.0')
# Provides a python function prototype for the c library function
arraytest.runsim.argtypes = [c_double, c_double, c_double, c_double, c_double]
# This is a wrapper function for the c library function
def runsim(k):
#Call the library function
	arraytest.runsim(k[i][0], k[i][1], k[i][2], k[i][3], k[i][4])

# Create a kvector - actually nonsense
k = np.ndarray(shape=(64,5), dtype=np.double)
#k = np.array([0.0, 0.0, 0.0, 20000, 0.0])

# Test the function
#k[0] = np.array([0.0, 0.0, 0.0, 10000, 0.0])

#print(k[0])
#     k[i][4] = k[i-1][4] + 1.0
a = (0.000997477)
GammaX = 2*math.pi/(a*math.sqrt(3))
GammaJ = 2*GammaX/math.sqrt(3)
XJ = GammaX/math.sqrt(3)

k[0] = np.array([0.0, 0, 0.0, 3000000.0, 0.0])
#k[0] = np.array([0.0, GammaJ, 0.0, 50000.0, 0.0])
step = ((GammaJ) + 2*XJ)/64

 #for i in range (1, 64):
#      k[i][0] = k[i-1][0]
#      k[i][1] = k[i-1][1] + step
#      k[i][2] = k[i-1][2]
#      k[i][3] = k[i-1][3] 
#      k[i][4] = k[i-1][4] + 1.0

for i in range (0, 1):
	p = "p%d" %i
	#print p
	p = multiprocessing.Process(target = runsim, args =(k,))
	#print i
	#print k[i]
	p.start()

# for i in range (kVal, (kVal + 5)):
# 	p = "p%d" %i
# 	print p
# 	p = multiprocessing.Process(target = runsim, args =(k,))
# 	print i
# 	print k[i]
# 	p.start()
# 	#    p.join()
# 	print "Done"
