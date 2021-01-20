#!/usr/local/apps/anaconda/3-2.2.0/bin/python
import time 
import numpy as np
import scipy as sp
import itertools as it
import math
import collections as cl
import matplotlib.pyplot as plt
import matplotlib as mpl

#______________________________________________________________________#
############################### Ey Field ###############################
#______________________________________________________________________#
print("Ey Fields")
for k in range(0,3):
	fig, axs = plt.subplots(2, 5, figsize = (18, 9),constrained_layout=True)
	Frame = []
	Nx = 105
	Ny = 181
	z = 0
	dt = 5.945e-18
	T = 2e6
	base = 6.0 + k*1.0
	print(base)
	for ax0 in axs.flat:
		lam = base + 0.10*z
		# Fieldx = "Raw/ExHeatMap%.2f.txt" %lam 
		Fieldy = "Raw/EyHeatMap%.2f.txt" %lam 
		# Fieldz = "Raw/EzHeatMap%.2f.txt" %lam 

		Full = np.zeros((Ny, Nx), dtype = np.double)
		# ExAb = np.zeros((Ny, Nx), dtype = np.double)
		EyAb = np.zeros((Ny, Nx), dtype = np.double)
		# EzAb = np.zeros((Ny, Nx), dtype = np.double)

		Y = np.linspace(0,  3.62, Ny)
		X = np.linspace(0,  2.10, Nx)

		# Ex = np.loadtxt(Fieldx, usecols=(0,), skiprows= 1, unpack =True )
		Ey = np.loadtxt(Fieldy, usecols=(0,), skiprows= 1, unpack =True )
		# Ez = np.loadtxt(Fieldz, usecols=(0,), skiprows= 1, unpack =True )
		# Ex = Ex*Ex*(dt/T)*(dt/T)
		# Ey = Ey*Ey*(dt/T)*(dt/T)
		# Ez = Ez*Ez*(dt/T)*(dt/T)

		Ey = (Ey)*(dt/T)
		print(max(Ey))
		print(min(Ey))

		# Ey = abs(Ey)*abs(Ey)*(dt/T)*(dt/T)
		# Ez = abs(Ez)*abs(Ez)*(dt/T)*(dt/T)

		for i in range (0, Ny):
			Full[i] = Ey[i*Nx: i*Nx + Nx]# + Ey[i*Nx: i*Nx + Nx]+ Ez[i*Nx: i*Nx + Nx]
			# Full[i-350] = np.sqrt(Full[i-350])
			# print(min(Full[i-350]))

		# print(max(np.sqrt(Ex)))
		
		norm = mpl.colors.Normalize(vmin=-2e-21, vmax=2e-21) 
		pc_kwargs = {'rasterized': True, 'cmap': 'jet'}

		im = ax0.pcolormesh(X, Y, Full, norm=norm, **pc_kwargs)
		ax0.set_aspect('equal')
		ax0.set_title(r"$\rm \lambda_{E_{z}} = %.2f \ \mu m $" %lam, fontsize = '25')

		ax0.set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '20')
		ax0.set_ylabel(r"$ \rm y \ (\mu m)$", fontsize = '20')
		plt.setp(ax0.spines.values(), linewidth=2)
		plt.tick_params(left = False, bottom = False)
		z = z + 1

	cbar = fig.colorbar(im, ax=axs)
	cbar.set_label(label = r"$\rm |E| \ \ Field \ (V m^{-1})$", size = '20')

	plt.savefig("Round/PDF/Ey2.10PlaneHM%.2f_%.2f_3_7Mag.pdf" %(base, lam))
	plt.savefig("Round/PNG/Ey2.10PlaneHM%.2f_%.2f_3_7Mag.png" %(base, lam))
#abs(Full*(dt/T))

#______________________________________________________________________#
############################### Ex Field ###############################
#______________________________________________________________________#
print("Ex Fields")
for k in range(0,3):
	fig, axs = plt.subplots(2, 5, figsize = (18, 9),constrained_layout=True)
	Frame = []
	Nx = 105
	Ny = 181
	z = 0
	dt = 5.945e-18
	T = 2e6
	base = 6.0 + k*1.0
	print(base)
	for ax0 in axs.flat:
		lam = base + 0.10*z
		Fieldx = "Raw/ExHeatMap%.2f.txt" %lam 
		# Fieldy = "Raw/ExHeatMap%.2f.txt" %lam 
		# Fieldz = "Raw/EzHeatMap%.2f.txt" %lam 

		Full = np.zeros((Ny, Nx), dtype = np.double)
		ExAb = np.zeros((Ny, Nx), dtype = np.double)
		# EyAb = np.zeros((Ny, Nx), dtype = np.double)
		# EzAb = np.zeros((Ny, Nx), dtype = np.double)

		Y = np.linspace(0,  3.62, Ny)
		X = np.linspace(0,  2.10, Nx)

		Ex = np.loadtxt(Fieldx, usecols=(0,), skiprows= 1, unpack =True )
		# Ey = np.loadtxt(Fieldy, usecols=(0,), skiprows= 1, unpack =True )
		# Ez = np.loadtxt(Fieldz, usecols=(0,), skiprows= 1, unpack =True )
		# Ex = Ex*Ex*(dt/T)*(dt/T)
		# Ey = Ey*Ey*(dt/T)*(dt/T)
		# Ez = Ez*Ez*(dt/T)*(dt/T)

		Ex = (Ex)*(dt/T)
		print(max(Ex))
		print(min(Ex))

		# Ey = abs(Ey)*abs(Ey)*(dt/T)*(dt/T)
		# Ez = abs(Ez)*abs(Ez)*(dt/T)*(dt/T)

		for i in range (0, Ny):
			Full[i] = Ex[i*Nx: i*Nx + Nx]# + Ey[i*Nx: i*Nx + Nx]+ Ez[i*Nx: i*Nx + Nx]
			# Full[i-350] = np.sqrt(Full[i-350])
			# print(min(Full[i-350]))

		# print(max(np.sqrt(Ex)))
		
		norm = mpl.colors.Normalize(vmin=-2e-21, vmax=2e-21) 
		pc_kwargs = {'rasterized': True, 'cmap': 'jet'}

		im = ax0.pcolormesh(X, Y, Full, norm=norm, **pc_kwargs)
		ax0.set_aspect('equal')
		ax0.set_title(r"$\rm \lambda_{E_{x}} = %.2f \ \mu m $" %lam, fontsize = '25')

		ax0.set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '20')
		ax0.set_ylabel(r"$ \rm Y \ (\mu m)$", fontsize = '20')
		plt.setp(ax0.spines.values(), linewidth=2)
		plt.tick_params(left = False, bottom = False)
		z = z + 1

	cbar = fig.colorbar(im, ax=axs)
	cbar.set_label(label = r"$\rm |E| \ \ Field \ (V m^{-1})$", size = '20')

	plt.savefig("Round/PDF/Ex2.10PlaneHM%.2f_%.2f_3_7Mag.pdf" %(base, lam))
	plt.savefig("Round/PNG/Ex2.10PlaneHM%.2f_%.2f_3_7Mag.png" %(base, lam))
#abs(Full*(dt/T))
#______________________________________________________________________#
############################### Ez Field ###############################
#______________________________________________________________________#
print("Ez Fields")
for k in range(0,3):
	fig, axs = plt.subplots(2, 5, figsize = (18, 9), constrained_layout=True)
	Frame = []
	Nx = 105
	Ny = 181
	z = 0
	dt = 5.945e-18
	T = 2e6
	base = 6.0 + k*1.0
	print(base)
	for ax0 in axs.flat:
		lam = base + 0.10*z
		# Fieldx = "Raw/ExHeatMap%.2f.txt" %lam 
		# Fieldy = "Raw/EzHeatMap%.2f.txt" %lam 
		Fieldz = "Raw/EzHeatMap%.2f.txt" %lam 

		Full = np.zeros((Ny, Nx), dtype = np.double)
		# ExAb = np.zeros((Ny, Nx), dtype = np.double)
		# EyAb = np.zeros((Ny, Nx), dtype = np.double)
		EzAb = np.zeros((Ny, Nx), dtype = np.double)

		Y = np.linspace(0,  3.62, Ny)
		X = np.linspace(0,  2.10, Nx)

		# Ex = np.loadtxt(Fieldx, usecols=(0,), skiprows= 1, unpack =True )
		# Ey = np.loadtxt(Fieldy, usecols=(0,), skiprows= 1, unpack =True )
		Ez = np.loadtxt(Fieldz, usecols=(0,), skiprows= 1, unpack =True )
		# Ex = Ex*Ex*(dt/T)*(dt/T)
		# Ey = Ey*Ey*(dt/T)*(dt/T)
		# Ez = Ez*Ez*(dt/T)*(dt/T)

		Ez = (Ez)*(dt/T)
		print(max(Ez))
		print(min(Ez))

		# Ey = abs(Ey)*abs(Ey)*(dt/T)*(dt/T)
		# Ez = abs(Ez)*abs(Ez)*(dt/T)*(dt/T)

		for i in range (0, Ny):
			Full[i] = Ez[i*Nx: i*Nx + Nx]# + Ey[i*Nx: i*Nx + Nx]+ Ez[i*Nx: i*Nx + Nx]
			# Full[i-350] = np.sqrt(Full[i-350])
			# print(min(Full[i-350]))

		# print(max(np.sqrt(Ex)))
		
		norm = mpl.colors.Normalize(vmin=-3e-21, vmax=3e-21) 
		pc_kwargs = {'rasterized': True, 'cmap': 'jet'}

		im = ax0.pcolormesh(X, Y, Full, norm=norm, **pc_kwargs)
		ax0.set_aspect('equal')
		ax0.set_title(r"$\rm \lambda_{E_{z}} = %.2f \ \mu m $" %lam, fontsize = '25')

		ax0.set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '20')
		ax0.set_ylabel(r"$ \rm Y \ (\mu m)$", fontsize = '20')
		plt.setp(ax0.spines.values(), linewidth=2)
		plt.tick_params(left = False, bottom = False)
		z = z + 1

	cbar = fig.colorbar(im, ax=axs)
	cbar.set_label(label = r"$\rm |E| \ \ Field \ (V m^{-1})$", size = '20')

	plt.savefig("Round/PDF/Ez2.10PlaneHM%.2f_%.2f_3_7Mag.pdf" %(base, lam))
	plt.savefig("Round/PNG/Ez2.10PlaneHM%.2f_%.2f_3_7Mag.png" %(base, lam))
#abs(Full*(dt/T))

#______________________________________________________________________#
############################## |E| Field ###############################
#______________________________________________________________________#
print("|E| Fields")
for k in range(0,3):
	fig, axs = plt.subplots(2, 5, figsize = (18, 9),constrained_layout=True)
	Frame = []
	Nx = 105
	Ny = 181
	z = 0
	dt = 5.945e-18
	T = 2e6
	base = 6.0 + k*1.0
	print(base)
	for ax0 in axs.flat:
		lam = base + 0.10*z
		Fieldx = "Raw/ExHeatMap%.2f.txt" %lam 
		Fieldy = "Raw/EyHeatMap%.2f.txt" %lam 
		Fieldz = "Raw/EzHeatMap%.2f.txt" %lam 

		Full = np.zeros((Ny, Nx), dtype = np.double)
		ExAb = np.zeros((Ny, Nx), dtype = np.double)
		EyAb = np.zeros((Ny, Nx), dtype = np.double)
		EzAb = np.zeros((Ny, Nx), dtype = np.double)

		Y = np.linspace(0,  3.62, Ny)
		X = np.linspace(0,  2.10, Nx)

		Ex = np.loadtxt(Fieldx, usecols=(0,), skiprows= 1, unpack =True )
		Ey = np.loadtxt(Fieldy, usecols=(0,), skiprows= 1, unpack =True )
		Ez = np.loadtxt(Fieldz, usecols=(0,), skiprows= 1, unpack =True )

		print(max(Ey))
		print(min(Ey))

		Ex = abs(Ex)*abs(Ex)*(dt/T)*(dt/T)
		Ey = abs(Ey)*abs(Ey)*(dt/T)*(dt/T)
		Ez = abs(Ez)*abs(Ez)*(dt/T)*(dt/T)

		for i in range (0, Ny):
			Full[i] = Ey[i*Nx: i*Nx + Nx] + Ey[i*Nx: i*Nx + Nx]+ Ez[i*Nx: i*Nx + Nx]
			Full[i] = np.sqrt(Full[i])

		# print(max(np.sqrt(Ex)))
		
		norm = mpl.colors.Normalize(vmin=0, vmax=3.5e-21) 
		pc_kwargs = {'rasterized': True, 'cmap': 'jet'}

		im = ax0.pcolormesh(X, Y, Full, norm=norm, **pc_kwargs)
		ax0.set_aspect('equal')
		ax0.set_title(r"$\rm \lambda_{|E|} = %.2f \ \mu m $" %lam, fontsize = '25')

		ax0.set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '20')
		ax0.set_ylabel(r"$ \rm Y \ (\mu m)$", fontsize = '20')
		plt.setp(ax0.spines.values(), linewidth=2)
		plt.tick_params(left = False, bottom = False)
		z = z + 1

	cbar = fig.colorbar(im, ax=axs)
	cbar.set_label(label = r"$\rm |E| \ \ Field \ (V m^{-1})$", size = '20')

	plt.savefig("Round/PDF/MagE_2.10PlaneHM%.2f_%.2f_3_7Mag.pdf" %(base, lam))
	plt.savefig("Round/PNG/MagE_2.10PlaneHM%.2f_%.2f_3_7Mag.png" %(base, lam))