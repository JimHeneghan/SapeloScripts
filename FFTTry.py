import time 
import numpy as np
import scipy as sp

import itertools 
import math
import collections as cl
import matplotlib.pyplot as plt
import matplotlib as mpl
#from mpl_toolkits.axes_grid1 import make_axes_locatable
# fig = plt.figure(1, figsize = (7, 8), constrained_layout=True)
# ax = fig.add_subplot(111)
# cmatch = ["c", "olive", "purple", "goldenrod", "orangered", "steelblue"]
shades = ['red', 'orange', 'gold', 'green', 'blue', 'purple', 'magenta']
freq = ['6.1', '6.3', '6.4', '6.5', '6.9', '7.0']
c0 = 3e8
threshold = 0
numpad = 100000
Eline = {}
lam1  = {}
lam2  = {}
lam3  = {}
lam4  = {}
lam5  = {}
lam6  = {}
MyLegend = []
# for z in range(6, 0):
z = 4
q = 0
Nx = 129
Ny = 1100
def flip(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])
for rr in range (0,5):
	fig, axs = plt.subplots(5,2, figsize = (24, 30),constrained_layout=True)
	mm = 0
	for ax0 in axs.flat:
		base = 6.30 + 0.02*mm + 0.2*rr # 6.0 + 0.1*sbox 
		print(base)
		print(rr)
		print(mm)
		Field = "Raw/EyHeatMap%.2f.txt" %base
		# ax = fig.add_subplot(111)
		# Field = "Raw/EyHeatMap6.80.txt"
		# z = z
		Full = np.zeros((Ny, Nx), dtype = np.double)
		Ey = np.loadtxt(Field, usecols=(0), skiprows= 1, unpack =True )

		print(Field)
		for i in range (0, Ny):
			Full[i] = Ey[i*Nx: i*Nx + Nx]

		E = Full[565] 
		x = np.linspace(0,  2.58e-6, Nx)

		padE = np.pad(E, numpad, mode='constant')
		# Efft = np.fft.fft(padE, len(padE))
		# fs = 1/(x[1]*len(padE))
		# f = np.arange(0, fs*len(padE), fs)
		# # lam = 1/f
		n = padE.size 

		dx = x[1]
		print(dx)
		dk = 1/(n*dx)
		k = np.arange(0,n*dk, dk)


		Efft = abs(np.fft.fft(padE))

		# Fourier transform data and take absolute value
		# Efft = abs(fft(pe))

		# print("Excitation Frequency", 1/(lam/1e4), "cm^-1")

		# Find peaks and store peak data
		counter = 0
		lam = []
		peakk = []
		peakEfft = []
		for i in range(2,n//2):
		    if ((Efft[i] > Efft[i-1]) & (Efft[i] > Efft[i+1]) & (Efft[i] > threshold)&(counter < 7)):
		        # print(Efft[i], 1.0/k[i], "um", 2*math.pi*k[i]*1e4, "cm-1")
		        peakk.append((2*math.pi)/(k[i]*1e-4))
		        peakEfft.append(Efft[i])
		        # print(counter)
		        if ((counter < 7)&(counter > 0)):
	        		lam.append(1e6/k[i])
		        	# print(lam[counter - 1])
	        	counter = counter + 1
		
		print(len(lam))
		# ax0.plot(k, Efft, linewidth=2, label = r"$ \rm %s \ \mu m , \ \lambda_{exciton} = %2.2f \ \mu m , \  %2.2f \ \mu m , \  %2.2f \ \mu m , \  %2.2f \ \mu m  $" %(freq[z], lam[0], lam[1], lam[2], lam[3]), color = shades[z])
		Eline[q], = ax0.plot(k, Efft, linewidth=2, color = shades[z])
		# if (q == 0):
			# handles, labels = ax0.get_legend_handles_labels()
		handles, labels = ax0.get_legend_handles_labels()
		#plt.savefig("EFieldXSec43THz.pdf")
		# ax0.axvline(x = -0.68, linestyle = "dashed", color = 'black')
		# ax0.axvline(x =  0.68, linestyle = "dashed", color = 'black')
		ax0.set_xlim(228000, 2e7)
		# ax0.set_ylim(0,1.1e-19)
		plt.setp(ax0.spines.values(), linewidth=1)
		plt.tick_params(left = False, bottom = False)
		ax0.set_xlabel(r"$ \rm q \ rad(cm^{-1})$", fontsize = '20')
		ax0.set_ylabel(r"$\rm s-SNOM \ Signal \ (arb. Units)$", fontsize = '20')
		ax0.set_title(r"$\rm \lambda_{E_{y}} = %.2f \ \mu m $" %base, fontsize = '20')


		ax0.scatter(peakk[0], peakEfft[0], s=25,edgecolors = 'black', c='r', zorder = 10, label = r"$%2.2f \ rad(cm^{-1})$" %((2*math.pi)/(lam[0]*1e-4)))		
		ax0.scatter(peakk[1], peakEfft[1], s=25,edgecolors = 'black', c='yellow', zorder = 10, label = r"$%2.2f \ rad(cm^{-1})$" %((2*math.pi)/(lam[1]*1e-4)))		
		ax0.scatter(peakk[2], peakEfft[2], s=25,edgecolors = 'black', c='lime', zorder = 10, label = r"$%2.2f \ rad(cm^{-1})$" %((2*math.pi)/(lam[2]*1e-4)))		
		ax0.scatter(peakk[3], peakEfft[3], s=25,edgecolors = 'black', c='cyan', zorder = 10, label = r"$%2.2f \ rad(cm^{-1})$" %((2*math.pi)/(lam[3]*1e-4)))		
		ax0.scatter(peakk[4], peakEfft[4], s=25,edgecolors = 'black', c='blue', zorder = 10, label = r"$%2.2f \ rad(cm^{-1})$" %((2*math.pi)/(lam[4]*1e-4)))		
		ax0.scatter(peakk[5], peakEfft[5], s=25,edgecolors = 'black', c='pink', zorder = 10, label = r"$%2.2f \ rad(cm^{-1})$" %((2*math.pi)/(lam[5]*1e-4)))		
		
		ax0.legend(ncol = 3,loc = 'upper right', fancybox= True, shadow = True, fontsize = 12) #bbox_to_anchor=(1.05, 0.5),

		mm = mm+1
		q = q + 1

	# for q in range (0, 6):
	# print(handles)
	# # sort both labels and handles by labels
	# labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
	temper = 6.30 + 0.2*rr
	plt.savefig("Lineouts/FFT%.2fpeak.png" %temper)
	# plt.savefig("6_9LineOutFFTNov17_5peak.pdf")