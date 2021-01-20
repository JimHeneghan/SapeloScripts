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
fig, axs = plt.subplots(2,3, figsize = (30, 18),constrained_layout=True)



z = 0
Y = np.linspace(0,  Ny, Ny)
X = np.linspace(-1.19,  1.19, Nx)
for ax0 in axs.flat:
	
	lam = 6.73 + 0.04*z # 6.0 + 0.1*sbox 
	Field = "Raw/EyHeatMap%.2f.txt" %lam 
	print(Field)
	Full = np.zeros((Ny, Nx), dtype = np.double)
	Ey = np.loadtxt(Field, usecols=(0), skiprows= 1, unpack =True )


	for i in range (0, Ny):
		Full[i] = Ey[i*Nx: i*Nx + Nx]

	ax0.plot(X, Full[565], linewidth=2, color = "black")

	peakx = []
	peakE = []

	for i in range(0,len(X)-1):
	    if ((Full[565,i] > Full[565,i-1]) & (Full[565,i] > Full[565,i+1])):
	        # print(Efft[i], 1.0/k[i], "um", 2*math.pi*k[i]*1e4, "cm-1")
	        peakx.append(X[i])
	        peakE.append(Full[565,i])

	# peakx = np.asarray(peakx)
	for i in range(0,len(peakx)):
		ax0.scatter(peakx[i], peakE[i], linewidth = 2, s=70,edgecolors = 'black', zorder = 25, c='gold', label = r"$ \rm %.2f \ \mu m$" %peakx[i])

	for i in range (0, len(peakx)):
		ax0.axvline(x =  peakx[i], linestyle = "dashed", color = 'black')

	ax0.set_title(r"$\rm \lambda_{E_{y}} = %.2f \ \mu m $" %lam, fontsize = '20')


	ax0.axvline(x =  -0.68, linestyle = "dashed", color = 'red', linewidth = 3)
	ax0.axvline(x =   0.68, linestyle = "dashed", color = 'red', linewidth = 3)

	ax0.set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '20')
	ax0.set_ylabel(r"$ \rm Intensity $", fontsize = '20')
	ax0.tick_params(direction = 'in', width=2, labelsize=15, bottom = False)
	# peakx = np.around(peakx, 2)
	# ax0.set_xticks(peakx)
	# ax0.set_xticklabels(peakx)
	plt.setp(ax0.spines.values(), linewidth=2)
	z = z+1
	if (z< 4):
		ax0.legend(fontsize = '15', bbox_to_anchor=(0., 1.12, 1., .102), loc='lower left',
           ncol=4, mode="expand", borderaxespad=0)
	else:
		ax0.legend(fontsize = '15', bbox_to_anchor=(0., -.35, 1., .102), loc='lower left',
           ncol=4, mode="expand", borderaxespad=0)

plt.savefig("Lineout6_73_6_93_CavLines.png") 
plt.clf()