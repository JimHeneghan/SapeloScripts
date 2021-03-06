import numpy as np
from scipy import *
from pylab import *
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

lam, n, k= np.loadtxt("PalikAg.txt", usecols=(1,2,3), skiprows= 0, unpack =True )

#using the equations in chapter 2.2.r of Macleod
#assuming the impedance of free space cancels out
#assuming the incident media is vacuum with k0 = 0
#The Ag nk data comes from an experiment performed on a 
#20 nm Ag film deposited on SiO2
# 
# unlabled equation on p 38 in Macleod after eqn 2.88 
n1 = n - 1j*k
d = 20e-9
imp0 = 376.730313
inc = math.pi*55/180.0
delta1 = n1*d*2*math.pi*cos(inc)/(lam*1e-6)

#Sellmeier formula for SiO2 from 
#https://refractiveindex.info/?shelf=main&book=SiO2&page=Malitson
# nb = np.sqrt(1- ((0.6961663*lam**2)/(lam**2 - 0.0684043**2)) + ((0.4079426*lam**2)/(lam**2 - 0.1162414**2)) + ((0.8974794*lam**2)/(lam**2 - 9.896161**2)))
# print(nb)
# eqn 2.93 in Macleod
#since we behin at normal incidence eta0 = y0
eta0 = imp0
eta1 = (n1)*imp0
eta2 = imp0
Y =  (eta2*cos(delta1) + 1j*eta1*sin(delta1))/(cos(delta1) + 1j*(eta2/eta1)*sin(delta1))

Rm = abs(((eta0 - Y)/(eta0 + Y))*conj((eta0 - Y)/(eta0 + Y)))

#Calculating the T

# Calculate (Power) Transmision from the result of problem 5.11 
# from: http://eceweb1.rutgers.edu/~orfanidi/ewa/ch05.pdf
# Note j --> -i Convention in formula below

B = cos(delta1) + 1j*(eta2/eta1)*sin(delta1)
C = eta2*cos(delta1) + 1j*eta1*sin(delta1)
Tm = 4*eta0*real(eta2)/((eta0*B + C)*conj(eta0*B + C))

#############################################################
#####################  Drude ################################
#############################################################
def func(lam0, wp, gamma, eps_inf):
	# nb = np.sqrt(1- ((0.6961663*lam0**2)/(lam0**2 - 0.0684043**2)) + ((0.4079426*lam0**2)/(lam0**2 - 0.1162414**2)) + ((0.8974794*lam0**2)/(lam0**2 - 9.896161**2)))

	c0 = 3e8
	w = c0/(lam0*1e-6)
	# k0 = k*1e12
	c0 = 3e8
	eps1 = eps_inf*(1 - (wp*wp)/(w*(w+1j*gamma)))
	n1 = np.sqrt(eps1)

	delta1 = n1*d*(w/c0)*2*math.pi*cos(inc)


	eta0 = imp0
	eta1 = (n1)*imp0
	eta2 = imp0
	Y =  (eta2*cos(delta1) + 1j*eta1*sin(delta1))/(cos(delta1) + 1j*(eta2/eta1)*sin(delta1))

	return abs(((eta0 - Y)/(eta0 + Y))*conj((eta0 - Y)/(eta0 + Y)))

# B = cos(delta1) + 1j*(eta2/eta1)*sin(delta1)
# C = eta2*cos(delta1) + 1j*eta1*sin(delta1)
# TDrude = 4*eta0*real(eta2)/((eta0*B + C)*conj(eta0*B + C))

#############################################################

popt, pcov = curve_fit(func, lam, Rm, p0 =[2.321e15, 5.513e12, 1])
print(popt)



fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
#plt.tight_layout(pad=5.4, w_pad=5.5, h_pad=5.4)
plt.setp(ax.spines.values(), linewidth=2)
tick_params(direction = 'in', width=2, labelsize=20)
# ylabel("T", fontsize = '30')   
ax.set_xlabel(r"$\rm Wavelength \ (\mu m)$", fontsize = '30')
# plt.xlim(0,4)
plt.plot(lam, Rm, 'o', markeredgecolor = "black", markerfacecolor = 'red')#, label = "Digitized Data")
ax.plot(lam, func(lam, *popt), color = "black", linewidth = 3, label = r"$\rm \gamma_{fit} = %.2f$ " %popt[0])# "\n" r"J = %f" %tuple(popt))
    
# plot(lam, Tm,  label=r'$\rm T_{nk: Ag}$', color='green')
# plot(lam, Rm + Tm,  label=r'$\rm R+T_{nk: Ag}$', color='blue')

plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(left = False, bottom = False)   
plt.legend(loc='center right', fontsize='30')
# ax.axvline(x = c0/(wp*1e-6), color = 'black', linewidth = 2)
plt.savefig("AgPalikDrudeFit.pdf")
plt.savefig("AgPalikDrudeFit.png")

# plt.show()
