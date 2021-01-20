import numpy as np
from scipy import *
from pylab import *
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
d = 50e-9
imp0 = 376.730313
# k0 = k*1e12
c0 = 3e8
numpad = 10000000 


Rk10 = np.loadtxt("K10/Ref/K40Apt24ExTraTime.txt", usecols=(0), skiprows= 1, unpack =True )
Rk20 = np.loadtxt("K20/Ref/K20Apt24ExTraTime.txt", usecols=(0), skiprows= 1, unpack =True )
Rk30 = np.loadtxt("K30/Ref/K30Apt24ExTraTime.txt", usecols=(0), skiprows= 1, unpack =True )
Rk40 = np.loadtxt("K40/Ref/K40Apt24ExTraTime.txt", usecols=(0), skiprows= 1, unpack =True )
Rk50 = np.loadtxt("K50/Ref/K50Apt24ExTraTime.txt", usecols=(0), skiprows= 1, unpack =True )
Rk70 = np.loadtxt("K70/Ref/K50Apt24ExTraTime.txt", usecols=(0), skiprows= 1, unpack =True )

# fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
# plt.setp(ax.spines.values(), linewidth=2)
# plt.tick_params(direction = 'in', width=2, labelsize=20)
# plt.ylabel("R", fontsize = '30')   
# plt.xlabel(r"$\rm \Delta t$", fontsize = '30')

# plt.setp(ax.spines.values(), linewidth=2)

# # plt.plot(Rk20, label = r"$\rm \kappa \ = \ 20$", color = "green")
# plt.plot(Rk30, label = r"$\rm \kappa \ = \ 30$", color = "blue")
# plt.plot(Rk10, label = r"$\rm \kappa \ = \ 10$", color = "red")

# plt.tick_params(left = False, bottom = False)   
# ax.legend(loc='upper center', fontsize='22')

# plt.savefig("RefKComps.png")
# plt.clf()

# Tk10 = np.loadtxt("K10/Ref/K10Apt24ExRefTime.txt", usecols=(0), skiprows= 1, unpack =True )
# # Rk20 = np.loadtxt("K20Apt24ExTraTime.txt", usecols=(0), skiprows= 1, unpack =True )
# Tk30 = np.loadtxt("K30/Ref/K30Apt24ExRefTime.txt", usecols=(0), skiprows= 1, unpack =True )

# fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
# plt.setp(ax.spines.values(), linewidth=2)
# plt.tick_params(direction = 'in', width=2, labelsize=20)
# plt.ylabel("R", fontsize = '30')   
# plt.xlabel(r"$\rm \Delta t$", fontsize = '30')

# plt.setp(ax.spines.values(), linewidth=2)

# # plt.plot(Rk20, label = r"$\rm \kappa \ = \ 20$", color = "green")
# plt.plot(Tk30, label = r"$\rm \kappa \ = \ 30$", color = "blue")
# plt.plot(Tk10, label = r"$\rm \kappa \ = \ 10$", color = "red")

# plt.tick_params(left = False, bottom = False)   
# ax.legend(loc='upper center', fontsize='22')

# plt.savefig("TraKComps.png")
# plt.clf()



fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(direction = 'in', width=2, labelsize=20)
plt.ylabel("R", fontsize = '30')   
plt.xlabel(r"$\rm \Delta t$", fontsize = '30')

plt.setp(ax.spines.values(), linewidth=2)

plt.plot(Rk10, label = r"$\rm \kappa \ = \ 10$", color = "lightgrey")
plt.plot(Rk20, label = r"$\rm \kappa \ = \ 20$", color = "silver")
plt.plot(Rk30, label = r"$\rm \kappa \ = \ 30$", color = "darkgray")
plt.plot(Rk40, label = r"$\rm \kappa \ = \ 40$", color = "dimgrey")
plt.plot(Rk50, label = r"$\rm \kappa \ = \ 50$", color = "black")
plt.plot(Rk70, label = r"$\rm \kappa \ = \ 70$", color = "red")



plt.tick_params(left = False, bottom = False)   
ax.legend(loc='upper center', fontsize='22')

plt.savefig("RefKCompsCoolColors.png")