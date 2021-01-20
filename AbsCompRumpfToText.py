import numpy as np
from scipy import *
from pylab import *
import math
import cmath
from ctypes import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib import colors as c
from scipy.optimize import curve_fit


cols     = ['red', 'darkgoldenrod', 'green', 'blue', 'purple']
litecols = ['lightpink',  'yellow', 'lime', 'cyan', 'magenta']

mpl.rcParams['agg.path.chunksize'] = 10000
#all units are in m^-1
##############################################################################
fig, ax0 = plt.subplots(figsize=(15,9),constrained_layout=True)
##############################################################################
dirk  = ['60', '105', '210', '490', '1020']
dirks = [60, 105, 210, 490, 1020]
gams  = [40.047, 27.2056, 9.622, 17.4781, 8.7441, 9.0982, 1.3777]
##############################################################################

dx     = 20e-9
dy     = 20e-9
c0     = 3e8
nref   = 1.0
ntra   = 3.42
NFREQs = 500

Nx = 47
Ny = 47
###########################################################################
lam        = np.loadtxt("105nm/Ref/2_38umPitchRTA.txt",  usecols=(0), skiprows= 1, unpack =True)

###########################################################################

###########################################################################
i = 0
j = 0
q = 0
for i in range(0,5):
	print(i)
	print(j)
	finder = "%snm/Ref/2_38umPitchRTA.txt" %dirk[i]
	Ref = np.loadtxt(finder,  usecols=(1), skiprows= 1, unpack =True)
	Tra = np.loadtxt(finder,  usecols=(2), skiprows= 1, unpack =True)
	Abs = np.loadtxt(finder,  usecols=(3), skiprows= 1, unpack =True)

	plt.setp(ax0.spines.values(), linewidth=2)
	ax0.tick_params(direction = 'in', width=2, labelsize=20)
	ax0.set_ylabel("A", fontsize = '30')   
	ax0.set_xlabel(r"$\rm Frequency \ (cm^{-1})$", fontsize = '30')
	ax0.set_xlim(1200,1700)
	ax0.set_ylim(0,1)


	ax0.plot(1/(lam*1e-4), Abs, label = r'$\rm A_{FDTD \ %s \ nm}$' %dirk[i], color = cols[i],
		linewidth = 8)
	# ax0.plot(1/(lam*1e-4), Tra, label = r'$\rm T_{%s}$' %dirk[q], color = "blue", linewidth = 3)
	# ax0.plot(1/(lam*1e-4), Abs, label = r'$\rm A_{%s}$' %dirk[q], color = "limegreen", linewidth = 3)

	ax0.tick_params(left = False, bottom = False)   
	ax0.yaxis.set_major_locator(MaxNLocator(nbins = 5,prune='lower'))


for i in range(0,5):

###########################################################################

	wp = 1.15136316e16
	gamma = 9.79125662e13
	eps_inf = 4.87
	s = 1.83
	omega_nu = 137200
	gamma = 100.0*gams[i]
	################################################
	eps_infz = 2.95
	sz = 0.61
	omega_nuz = 74606.285
	gammaz = 491.998
	################################################
	d = dirks[i]*1e-9
	imp0 = 376.730313
	k00 = linspace (50000, 2500000, 20000)
	c0 = 3e8
	eps1 = eps_inf + (s*(omega_nu**2))/((omega_nu**2) + 1j*gamma*k00 - k00*k00)
	eps1z = eps_infz + (sz*(omega_nuz**2))/((omega_nuz**2) + 1j*gammaz*k00 - k00*k00)


	n1 = np.sqrt(eps1)
	n1z = np.sqrt(eps1z)

	#using the equations in chapter 2.2.r of Macleod
	#assuming the impedance of free space cancels out
	#assuming the incident media is vacuum with k00 = 0

	# unlabled equation on p 38 in Macleod after eqn 2.88 
	delta1 = n1*d*k00*2*math.pi
	delta1z = n1z*d*k00*2*math.pi


	# eqn 2.93 in Macleod
	#since we behin at normal incidence eta0 = y0
	eta0 = imp0
	eta1 = (n1)*imp0
	eta2 = imp0
	Y =  (eta2*cos(delta1) + 1j*eta1*sin(delta1))/(cos(delta1) + 1j*(eta2/eta1)*sin(delta1))

	Rm = abs(((eta0 - Y)/(eta0 + Y))*conj((eta0 - Y)/(eta0 + Y)))

	#########################################################
	# eta0z = imp0
	# eta1z = (n1z)*imp0
	# eta2z = imp0
	# Yz =  (eta2z*cos(delta1z) + 1j*eta1z*sin(delta1z))/(cos(delta1z) + 1j*(eta2z/eta1z)*sin(delta1z))

	# Rmz = abs(((eta0z - Yz)/(eta0z + Yz))*conj((eta0z - Yz)/(eta0z + Yz)))
	#########################################################

	#Calculating the T

	# Calculate (Power) Transmision from the result of problem 5.11 
	# from: http://eceweb1.rutgers.edu/~orfanidi/ewa/ch05.pdf
	# Note j --> -i Convention in formula below

	B = cos(delta1) + 1j*(eta2/eta1)*sin(delta1)
	C = eta2*cos(delta1) + 1j*eta1*sin(delta1)
	Tm = 4*eta0*real(eta2)/((eta0*B + C)*conj(eta0*B + C))
#################################################################
	ax0.plot(k00/100, (1-(Rm+Tm)), label=r'$\rm A_{TM \ %s \ nm}$' %dirks[i], color=litecols[i], linewidth = 4)

ax0.legend(loc='center right', fontsize='15', ncol = 2)
plt.savefig("AbsSweep.png")

