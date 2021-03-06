#!/usr/local/apps/anaconda/3-2.2.0/bin/python
import time 
import numpy as np
import scipy as sp
import itertools as it
import math
import collections as cl
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

# fig, ((axs[z,q]0, axs[z,q]1, axs[z,q]2), (ax10, ax11, ax12), (ax20, ax21, ax22), (ax30, ax31, ax32)) = plt.subplots(4,3, figsize = (16,20))#, constrained_layout=True)
freq = []
selectionBox = [144, 140, 135]

# #______________________________________________________________________#
# ############################## Ex Field ################################
# #______________________________________________________________________#
# fig, axs = plt.subplots(2, 5, figsize = (18, 9),constrained_layout=True)
fig, axs = plt.subplots(6,4, figsize = (65,55))
# fig.subplots_adjust(hspace=0.25)

plt.gcf().subplots_adjust(left = 0.05, right = 0.95, top = 0.97, bottom = 0.03)

Frame = []
Nx = 119
Ny = 1100
z = 0
dt = 5.945e-18
T = 2e6
#print(z)
q = 0
for l in range(0,3):
	z  = 2*l
	print(z)
	sbox = selectionBox[l]
	lam = 6.0 + 0.01*sbox # 6.0 + 0.1*sbox 
	FieldFull = "Raw/ExHeatMap%.2f.txt" %lam 
	FieldDead = "Raw/ExHeatMap%.2f.txt" %lam#"../../../XSecHeatMap_Dead/2_38um/HeatMap/Raw/ExHeatMap%.2f.txt" %lam 

	print(FieldFull)



	Full = np.zeros((400, Nx), dtype = np.double)
	Dead = np.zeros((400, Nx), dtype = np.double)


	Y = np.linspace(0, 0.8, 400)
	X = np.linspace(-1.19, 1.19, Nx)

	EFull = np.loadtxt(FieldFull, usecols=(0,), skiprows= 1, unpack =True )
	EDead = np.loadtxt(FieldDead, usecols=(0,), skiprows= 1, unpack =True )

	ExFull = (EFull)*(dt/T)
	ExDead = (EDead)*(dt/T)
	for i in range (350, 750):
		Full[i-350] = ExFull[i*Nx: i*Nx + Nx]
		Dead[i-350] = ExDead[i*Nx: i*Nx + Nx]

	# print(max(E)*(dt/T))
	norm = mpl.colors.Normalize(vmin=-5e-21, vmax=5e-21)
	pc_kwargs = {'rasterized': True, 'cmap': 'jet'}

	im = axs[z,q].pcolormesh(X, Y, Full, norm=norm, **pc_kwargs)

	axs[z,q].set_title(r"$\rm Coupled \ Device \ \mathit{E_{x}} : \ \nu= %2.2f \ cm^{-1} $" %(1/(lam*1e-4)), fontsize = '50', pad = 15)
	axs[z,q].set_aspect('equal')

	axs[z,q].set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '50')
	axs[z,q].set_ylabel(r"$ \rm z \ (\mu m)$", fontsize = '50')
	plt.setp(axs[z,q].spines.values(), linewidth=2)
	axs[z,q].tick_params(left = False, bottom = False, labelsize = "20")

	cbar = fig.colorbar(im, ax=axs[z,q], orientation = 'horizontal')
	cbar.set_label(label = r"$\rm Ex \ \ Field \ (V m^{-1})$", size = '50')
	cbar.ax.tick_params(labelsize = 20)

	########################## Ex Dead ##############################
	z = z + 1
	print(z)
	im = axs[z,q].pcolormesh(X, Y, Dead, norm=norm, **pc_kwargs)

	axs[z,q].set_title(r"$\rm Uncoupled \ Device \ \mathit{E_{x}}  : \ \nu= %2.2f \ cm^{-1} $" %(1/(lam*1e-4)), fontsize = '50', pad = 15)
	axs[z,q].set_aspect('equal')

	axs[z,q].set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '50')
	axs[z,q].set_ylabel(r"$ \rm z \ (\mu m)$", fontsize = '50')
	plt.setp(axs[z,q].spines.values(), linewidth=2)
	axs[z,q].tick_params(left = False, bottom = False, labelsize = "20")
	cbar = fig.colorbar(im, ax=axs[z,q], orientation = 'horizontal')
	cbar.set_label(label = r"$\rm Ex \ \ Field \ (V m^{-1})$", size = '50')
	cbar.ax.tick_params(labelsize = 20)


	# plt.show()
# ax1_divider = make_axes_locatable(axs[z,q])
# cax1 = ax1_divider.append_axes("bottom", size="20%", pad="55%")


# cbar = fig.colorbar(im, cax=cax1, orientation = 'horizontal')
# cbar.set_label(label = r"$\rm Ex \ \ Field \ (V m^{-1})$", size = '50')
# cbar.ax.tick_params(labelsize = 15)


# plt.savefig("FullDev2_38um1ExHMs.pdf")
# plt.savefig("FullDev2_38um1ExHMs.png")

print("\n \n")


# #______________________________________________________________________#
# ############################## Ey Field ################################
# #______________________________________________________________________#


Frame = []
Nx = 119
Ny = 1100
# z = 0
dt = 5.945e-18
T = 2e6
q = q + 1
#print(z)
for l in range(0,3):
	z  = 2*l
	print(z)
	sbox = selectionBox[l]
	lam = 6.0 + 0.01*sbox # freq[sbox]
	FieldFull = "Raw/EyHeatMap%.2f.txt" %lam 
	FieldDead = "Raw/EyHeatMap%.2f.txt" %lam#"../../../XSecHeatMap_Dead/2_38um/HeatMap/Raw/EyHeatMap%.2f.txt" %lam

	Full = np.zeros((400, Nx), dtype = np.double)
	Dead = np.zeros((400, Nx), dtype = np.double)


	Y = np.linspace(0, 0.8, 400)
	X = np.linspace(-1.19, 1.19, Nx)

	EFull = np.loadtxt(FieldFull, usecols=(0,), skiprows= 1, unpack =True )
	EDead = np.loadtxt(FieldDead, usecols=(0,), skiprows= 1, unpack =True )

	EyFull = (EFull)*(dt/T)
	EyDead = (EDead)*(dt/T)
	for i in range (350, 750):
		Full[i-350] = EyFull[i*Nx: i*Nx + Nx]
		Dead[i-350] = EyDead[i*Nx: i*Nx + Nx]

	norm = mpl.colors.Normalize(vmin=-1e-21, vmax=1e-21) 
	pc_kwargs = {'rasterized': True, 'cmap': 'jet'}

	im = axs[z,q].pcolormesh(X, Y, Full, norm=norm, **pc_kwargs)

	axs[z,q].set_title(r"$\rm Coupled \ Device \ \mathit{E_{y}} : \ \nu= %2.2f \ cm^{-1} $" %(1/(lam*1e-4)), fontsize = '50', pad = 15)
	axs[z,q].set_aspect('equal')

	axs[z,q].set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '50')
	axs[z,q].set_ylabel(r"$ \rm y \ (\mu m)$", fontsize = '50')
	plt.setp(axs[z,q].spines.values(), linewidth=2)
	axs[z,q].tick_params(left = False, bottom = False, labelsize = "20")
	cbar = fig.colorbar(im, ax=axs[z,q], orientation = 'horizontal')
	cbar.set_label(label = r"$\rm Ey \ \ Field \ (V m^{-1})$", size = '50')
	cbar.ax.tick_params(labelsize = 20)
	########################## Ey Dead ##############################
	z = z + 1
	im = axs[z,q].pcolormesh(X, Y, Dead, norm=norm, **pc_kwargs)

	axs[z,q].set_title(r"$\rm Uncoupled \ Device \ \mathit{E_{y}} : \ \nu= %2.2f \ cm^{-1} $" %(1/(lam*1e-4)), fontsize = '50', pad = 15)
	axs[z,q].set_aspect('equal')

	axs[z,q].set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '50')
	axs[z,q].set_ylabel(r"$ \rm z \ (\mu m)$", fontsize = '50')
	plt.setp(axs[z,q].spines.values(), linewidth=2)
	axs[z,q].tick_params(left = False, bottom = False, labelsize = "20")
	cbar = fig.colorbar(im, ax=axs[z,q], orientation = 'horizontal')
	cbar.set_label(label = r"$\rm Ey \ \ Field \ (V m^{-1})$", size = '50')
	cbar.ax.tick_params(labelsize = 20)


print("\n \n")

# ax1_divider = make_axes_locatable(axs[z,q])
# cax1 = ax1_divider.append_axes("bottom", size="20%", pad="55%")
# cbar = fig.colorbar(im, cax=cax1, orientation = 'horizontal')
# cbar.set_label(label = r"$\rm Ey \ \ Field \ (V m^{-1})$", size = '50')
# cbar.ax.tick_params(labelsize = 15)
# plt.savefig("FullDev2_38umEyHMs.pdf")

# print("\n \n")

# # #______________________________________________________________________#
# # ############################## Ez Selec ################################
# # #______________________________________________________________________#


Frame = []
z = 0
dt = 5.945e-18
T = 2e6
q = q + 1
#print(z)
for l in range(0,3):
	z  = 2*l
	print(z)
	sbox = selectionBox[l]
	lam = 6.0 + 0.01*sbox # freq[sbox]
	FieldFull = "Raw/EzHeatMap%.2f.txt" %lam 
	FieldDead = "Raw/EzHeatMap%.2f.txt" %lam#"../../../XSecHeatMap_Dead/2_38um/HeatMap/Raw/EzHeatMap%.2f.txt" %lam
	# print(Field)



	Full = np.zeros((400, Nx), dtype = np.double)
	Dead = np.zeros((400, Nx), dtype = np.double)


	Y = np.linspace(0, 0.8, 400)
	X = np.linspace(-1.19, 1.19, Nx)

	EFull = np.loadtxt(FieldFull, usecols=(0,), skiprows= 1, unpack =True )
	EDead = np.loadtxt(FieldDead, usecols=(0,), skiprows= 1, unpack =True )

	EzFull = (EFull)*(dt/T)
	EzDead = (EDead)*(dt/T)
	for i in range (350, 750):
		Full[i-350] = EzFull[i*Nx: i*Nx + Nx]
		Dead[i-350] = EzDead[i*Nx: i*Nx + Nx]

	norm = mpl.colors.Normalize(vmin=-3e-21, vmax=3e-21)
	pc_kwargs = {'rasterized': True, 'cmap': 'jet'}

	im = axs[z,q].pcolormesh(X, Y, Full, norm=norm, **pc_kwargs)

	axs[z,q].set_title(r"$\rm Coupled \ Device \ \mathit{E_{z}} : \ \nu= %2.2f \ cm^{-1} $" %(1/(lam*1e-4)), fontsize = '50', pad = 15)
	axs[z,q].set_aspect('equal')

	axs[z,q].set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '50')
	axs[z,q].set_ylabel(r"$ \rm y \ (\mu m)$", fontsize = '50')
	plt.setp(axs[z,q].spines.values(), linewidth=2)
	axs[z,q].tick_params(left = False, bottom = False, labelsize = "20")
	cbar = fig.colorbar(im, ax=axs[z,q], orientation = 'horizontal')
	cbar.set_label(label = r"$\rm Ez \ \ Field \ (V m^{-1})$", size = '50')
	cbar.ax.tick_params(labelsize = 20)
	########################## Ez Dead ##############################
	z = z + 1
	im = axs[z,q].pcolormesh(X, Y, Dead, norm=norm, **pc_kwargs)

	axs[z,q].set_title(r"$\rm Uncoupled \ Device \ \mathit{E_{z}} : \ \nu= %2.2f \ cm^{-1} $" %(1/(lam*1e-4)), fontsize = '50', pad = 15)
	axs[z,q].set_aspect('equal')

	axs[z,q].set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '50')
	axs[z,q].set_ylabel(r"$ \rm z \ (\mu m)$", fontsize = '50')
	plt.setp(axs[z,q].spines.values(), linewidth=2)
	axs[z,q].tick_params(left = False, bottom = False, labelsize = "20")

	cbar = fig.colorbar(im, ax=axs[z,q], orientation = 'horizontal')
	cbar.set_label(label = r"$\rm Ez \ \ Field \ (V m^{-1})$", size = '50')
	cbar.ax.tick_params(labelsize = 20)

# ax1_divider = make_axes_locatable(axs[z,q])
# cax1 = ax1_divider.append_axes("bottom", size="20%", pad="55%")
# cbar = fig.colorbar(im, cax=cax1, orientation = 'horizontal')

# cbar.set_label(label = r"$\rm Ez \ \ Field \ (V m^{-1})$", size = '50')
# cbar.ax.tick_params(labelsize = 15)
# # plt.savefig("FullDev2_38umEzHMs.pdf")
# # plt.savefig("FullDev2_38umEzHMs.png")

# print("\n \n")




# # #______________________________________________________________________#
# # ############################## |E| Field ###############################
# # #______________________________________________________________________#


Frame = []

z = 0
dt = 5.945e-18
T = 2e6
q = q + 1

#print(z)
for l in range(0,3):
	z  = 2*l
	print(z)
	sbox = selectionBox[l]
	lam = 6.0 + 0.01*sbox # freq[sbox]
	Fieldx = "Raw/ExHeatMap%.2f.txt" %lam 
	Fieldy = "Raw/EyHeatMap%.2f.txt" %lam 
	Fieldz = "Raw/EzHeatMap%.2f.txt" %lam 

	FieldDeadx = "Raw/ExHeatMap%.2f.txt" %lam #"../../../XSecHeatMap_Dead/2_38um/HeatMap/Raw/ExHeatMap%.2f.txt" %lam
	FieldDeady = "Raw/EyHeatMap%.2f.txt" %lam #"../../../XSecHeatMap_Dead/2_38um/HeatMap/Raw/EyHeatMap%.2f.txt" %lam
	FieldDeadz = "Raw/EzHeatMap%.2f.txt" %lam #"../../../XSecHeatMap_Dead/2_38um/HeatMap/Raw/EzHeatMap%.2f.txt" %lam
	Full = np.zeros((400, Nx), dtype = np.double)
	Dead = np.zeros((400, Nx), dtype = np.double)


	Y = np.linspace(0, 0.8, 400)
	X = np.linspace(-1.19, 1.19, Nx)

	ExFull = np.loadtxt(Fieldx, usecols=(0,), skiprows= 1, unpack =True )
	EyFull = np.loadtxt(Fieldy, usecols=(0,), skiprows= 1, unpack =True )
	EzFull = np.loadtxt(Fieldz, usecols=(0,), skiprows= 1, unpack =True )

	ExDead = np.loadtxt(FieldDeadx, usecols=(0,), skiprows= 1, unpack =True )
	EyDead = np.loadtxt(FieldDeady, usecols=(0,), skiprows= 1, unpack =True )
	EzDead = np.loadtxt(FieldDeadz, usecols=(0,), skiprows= 1, unpack =True )


	ExFull = (EFull)*(dt/T)
	ExDead = (EDead)*(dt/T)
	for i in range (350, 750):
		Full[i-350] = ExFull[i*Nx: i*Nx + Nx]
		Dead[i-350] = ExDead[i*Nx: i*Nx + Nx]

	ExFull = abs(ExFull)*abs(ExFull)*(dt/T)*(dt/T)
	EyFull = abs(EyFull)*abs(EyFull)*(dt/T)*(dt/T)
	EzFull = abs(EzFull)*abs(EzFull)*(dt/T)*(dt/T)

	ExDead = abs(ExDead)*abs(ExDead)*(dt/T)*(dt/T)
	EyDead = abs(EyDead)*abs(EyDead)*(dt/T)*(dt/T)
	EzDead = abs(EzDead)*abs(EzDead)*(dt/T)*(dt/T)

	for i in range (350, 750):
		Full[i-350] = ExFull[i*Nx: i*Nx + Nx] + EyFull[i*Nx: i*Nx + Nx]+ EzFull[i*Nx: i*Nx + Nx]
		Full[i-350] = np.sqrt(Full[i-350])

		Dead[i-350] = ExDead[i*Nx: i*Nx + Nx] + EyDead[i*Nx: i*Nx + Nx]+ EzDead[i*Nx: i*Nx + Nx]
		Dead[i-350] = np.sqrt(Dead[i-350])
	
	norm = mpl.colors.Normalize(vmin=0, vmax=2.5e-21) 
	pc_kwargs = {'rasterized': True, 'cmap': 'jet'}

	im = axs[z,q].pcolormesh(X, Y, Full, norm=norm, **pc_kwargs)

	axs[z,q].set_title(r"$\rm Coupled \ Device \ \mathit{|E|} : \ \nu= %2.2f \ cm^{-1} $" %(1/(lam*1e-4)), fontsize = '50', pad = 15)
	axs[z,q].set_aspect('equal')

	axs[z,q].set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '50')
	axs[z,q].set_ylabel(r"$ \rm y \ (\mu m)$", fontsize = '50')
	plt.setp(axs[z,q].spines.values(), linewidth=2)
	axs[z,q].tick_params(left = False, bottom = False, labelsize = "20")
	cbar = fig.colorbar(im, ax=axs[z,q], orientation = 'horizontal')
	cbar.set_label(label = r"$\rm |E| \ \ Field \ (V m^{-1})$", size = '50')
	cbar.ax.tick_params(labelsize = 20)
	########################## |E| Dead ##############################
	z = z + 1

	im = axs[z,q].pcolormesh(X, Y, Dead, norm=norm, **pc_kwargs)

	axs[z,q].set_title(r"$\rm Uncoupled \ Device \ \mathit{|E|}  : \ \nu= %2.2f \ cm^{-1} $" %(1/(lam*1e-4)), fontsize = '50', pad = 15)
	axs[z,q].set_aspect('equal')

	axs[z,q].set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '50')
	axs[z,q].set_ylabel(r"$ \rm z \ (\mu m)$", fontsize = '50')
	plt.setp(axs[z,q].spines.values(), linewidth=2)
	axs[z,q].tick_params(left = False, bottom = False, labelsize = "20")
	cbar = fig.colorbar(im, ax=axs[z,q], orientation = 'horizontal')
	cbar.set_label(label = r"$\rm |E| \ \ Field \ (V m^{-1})$", size = '50')
	cbar.ax.tick_params(labelsize = 20)

# ax1_divider = make_axes_locatable(axs[z,q])
# cax1 = ax1_divider.append_axes("bottom", size="20%", pad="55%")
# cbar = fig.colorbar(im, cax=cax1, orientation = 'horizontal')

# cbar.set_label(label = r"$\rm |E| \ \ Field \ (V m^{-1})$", size = '50')
# cbar.ax.tick_params(labelsize = 15)
# plt.savefig("FullDev2_38um2MagEHMs.pdf")
# plt.savefig("FullDev2_38um2MagEHMs.png")
plt.savefig("Asymptote_2_38um.png")

print("\n \n")
