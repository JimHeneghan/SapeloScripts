from pylab import *
from scipy.optimize import curve_fit

dcc, freq1, freq2 = loadtxt("freqvsdcc2.txt",  usecols=(0,1,2), skiprows= 0, unpack =True)
plot(dcc,freq1, 'o')
plot(dcc,freq2, 'o')
hlines(freq2[0],dcc[0],dcc[-1],color='green')
show()