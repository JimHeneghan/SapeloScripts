import time 
import numpy as np
import scipy as sp
import itertools as it
import math
import collections as cl
import matplotlib.pyplot as plt
import matplotlib as mpl
rain = ['red', 'orange', 'gold', 'green', 'blue', 'purple', 'magenta', 'red', 'orange', 'gold', 'green', 'blue', 'purple', 'magenta']
Nx = 119
Ny = 1100

lineOut = np.zeros((300, Nx), dtype = np.double)
wl      = np.zeros(300, dtype = np.double)
for k in range (0,300):
	lam = 6.30 + 0.01*k # 6.0 + 0.1*sbox 
	wl[k] = lam
	Field = "Raw/EyHeatMap%.2f.txt" %lam 
	print(Field)
	Full = np.zeros((Ny, Nx), dtype = np.double)
	Ey = np.loadtxt(Field, usecols=(0), skiprows= 1, unpack =True )


	for i in range (0, Ny):
		Full[i] = Ey[i*Nx: i*Nx + Nx]

	lineOut[k] = Full[565]


# 	plt.clf()
# plt.clf()

# c0 = 3e8
# threshold = 0.02
# numpad = 100000
# Eline = {}
# lam1  = {}
# lam2  = {}
# lam3  = {}
# lam4  = {}
# MyLegend = []
# def flip(items, ncol):
#     return itertools.chain(*[items[i::ncol] for i in range(ncol)])

# for rr in range (0,5):
# 	fig, axs = plt.subplots(5,2, figsize = (24, 30),constrained_layout=True)

# 	Y = np.linspace(0,  Ny, Ny)
# 	X = np.linspace(-1.19,  1.19, Nx)
# 	q = 0
# 	mm = 0
# 	for ax0 in axs.flat:
# 		# ax = fig.add_subplot(111)

# 		temp2 = 6.30 + 0.02*mm + 0.2*rr # 6.0 + 0.1*sbox 
# 		print(temp2)
# 		Field = "Raw/EyHeatMap%.2f.txt" %temp2 
# 		print(Field)
# 		Full = np.zeros((Ny, Nx), dtype = np.double)
# 		x, E = np.loadtxt(Field, usecols=(0,1), skiprows= 3, unpack =True )

# 		padE = np.pad(E, numpad, mode='constant')
# 		# Efft = np.fft.fft(padE, len(padE))
# 		# fs = 1/(x[1]*len(padE))
# 		# f = np.arange(0, fs*len(padE), fs)
# 		# # lam = 1/f
# 		n = padE.size 
# 		dx = x[1]
# 		dk = 1/(n*dx)
# 		k = np.arange(0,n*dk, dk)


# 		Efft = abs(np.fft.fft(padE))

# 		# Fourier transform data and take absolute value
# 		# Efft = abs(fft(pe))

# 		# print("Excitation Frequency", 1/(lam/1e4), "cm^-1")

# 		# Find peaks and store peak data
# 		counter = 0
# 		lam = []
# 		peakk = []
# 		peakEfft = []
# 		for i in range(2,n//2):
# 		    if ((Efft[i] > Efft[i-1]) & (Efft[i] > Efft[i+1]) & (Efft[i] > threshold)&(counter < 5)):
# 		        print(Efft[i], 1.0/k[i], "um", 2*math.pi*k[i]*1e4, "cm-1")
# 		        peakk.append(k[i])
# 		        peakEfft.append(Efft[i])
# 		        # print(counter)
# 		        if ((counter < 5)&(counter > 0)):
# 		        	lam.append(1e6/k[i])
# 		        	# print(lam[counter - 1])
# 	        	counter = counter + 1
		
# 		Eline[q], = ax0.plot(k, Efft, linewidth=2, label = r"$ s-SNOM$", color = 'black')
# 		#plt.savefig("EFieldXSec43THz.pdf")
# 		# ax0.axvline(x = -0.68, linestyle = "dashed", color = 'black')
# 		# ax0.axvline(x =  0.68, linestyle = "dashed", color = 'black')
# 		# ax0.set_xlim(0, 5e6)
# 		# ax0.set_ylim(0,5)
# 		plt.setp(ax0.spines.values(), linewidth=1)
# 		plt.tick_params(left = False, bottom = False)
# 		ax0.set_xlabel(r"$ \rm k \ (m^{-1})$", fontsize = '35')
# 		ax0.set_ylabel(r"$\rm s-SNOM \ Signal \ (arb. units)$", fontsize = '20')

# 		lam1[q] =ax0.scatter(peakk[0], peakEfft[0], s=25,edgecolors = 'black', c='r', zorder = 10, label = r"$%2.2f \ \mu m , \ $" %lam[0])		
# 		lam2[q] =ax0.scatter(peakk[1], peakEfft[1], s=25,edgecolors = 'black', c='yellow', zorder = 10, label = r"$%2.2f \ \mu m , \ $" %lam[1])		
# 		lam3[q] =ax0.scatter(peakk[2], peakEfft[2], s=25,edgecolors = 'black', c='lime', zorder = 10, label = r"$%2.2f \ \mu m , \ $" %lam[2])		
# 		lam4[q] =ax0.scatter(peakk[3], peakEfft[3], s=25,edgecolors = 'black', c='cyan', zorder = 10, label = r"$%2.2f \ \mu m$" %lam[3])		

# 		ax0.legend(loc = 'center right', fancybox= True, shadow = True, fontsize = 12) #bbox_to_anchor=(1.05, 0.5),

# 		q = q + 1
# 		mm = mm + 1
# 		# for q in range (0, 6):
# 		# 	handles.append(Eline[q])
# 		# for q in range (0, 1):
# 		# 	handles.append(lam1[q])
# 		# 	# handler_map={tuple: mpl.legend_handler.HandlerTuple(ndivide=None)}
# 		# 	# temp = lam1[q]
# 		# 	# labels.append(temp.get_label)
# 		# for q in range (0, 1):
# 		# 	handles.append(lam2[q])
# 		# for q in range (0, 1):
# 		# 	handles.append(lam3[q])
# 		# for q in range (0, 1):
# 		# 	handles.append(lam4[q])

# # 		# # print
# # 	temper = 6.30 + 0.2*k

# 	plt.savefig("Lineouts/FFT%d_2_58dcc.png" %rr)
# 	plt.clf()
