import numpy as np
from scipy import *
from pylab import *
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

lam, n, k= np.loadtxt("PalikAg.txt", usecols=(1,3,4), skiprows= 1, unpack =True )
lam = lam/100
#using the equations in chapter 2.2.r of Macleod
#assuming the impedance of free space cancels out
#assuming the incident media is vacuum with k0 = 0
#The Ag nk data comes from an experiment performed on a 
#20 nm Ag film deposited on SiO2
# 
# unlabled equation on p 38 in Macleod after eqn 2.88 
N1 = n - 1j*k
d = 20e-9
imp0 = 376.730313



#############################################################
##################### Real Drude ############################
#############################################################
def func_Re(lam0, wp, gamma, eps_inf):
	# nb = np.sqrt(1- ((0.6961663*lam0**2)/(lam0**2 - 0.0684043**2)) + ((0.4079426*lam0**2)/(lam0**2 - 0.1162414**2)) + ((0.8974794*lam0**2)/(lam0**2 - 9.896161**2)))

	c0 = 3e8
	w = c0/(lam0*1e-6)
	# k0 = k*1e12
	c0 = 3e8
	eps1 = eps_inf*(1 - (wp*wp)/(w*(w+1j*gamma)))
	N2 = np.sqrt(eps1)

	


	return abs(N2)

# B = cos(delta1) + 1j*(eta2/eta1)*sin(delta1)
# C = eta2*cos(delta1) + 1j*eta1*sin(delta1)
# TDrude = 4*eta0*real(eta2)/((eta0*B + C)*conj(eta0*B + C))

#############################################################

popt_Re, pcov_Re = curve_fit(func_Re, lam, abs(N1), p0 =[2.321e15, 5.513e12, 1])
print(popt_Re)


# #############################################################
# ##################### Imaginary Drude ########################
# #############################################################
# def func_Im(lam0, wp, gamma, eps_inf):
# 	# nb = np.sqrt(1- ((0.6961663*lam0**2)/(lam0**2 - 0.0684043**2)) + ((0.4079426*lam0**2)/(lam0**2 - 0.1162414**2)) + ((0.8974794*lam0**2)/(lam0**2 - 9.896161**2)))

# 	c0 = 3e8
# 	w = c0/(lam0*1e-6)
# 	# k0 = k*1e12
# 	c0 = 3e8
# 	eps1 = eps_inf*(1 - (wp*wp)/(w*(w-1j*gamma)))
# 	N2 = np.sqrt(eps1)

	


# 	return N2.imag

# # B = cos(delta1) + 1j*(eta2/eta1)*sin(delta1)
# # C = eta2*cos(delta1) + 1j*eta1*sin(delta1)
# # TDrude = 4*eta0*real(eta2)/((eta0*B + C)*conj(eta0*B + C))

# #############################################################

# popt_Im, pcov_Im = curve_fit(func_Im, lam, k)#, p0 =[2.321e15, 5.513e12, 1])
# print(popt_Im)

fig, ax = plt.subplots(figsize=(8,9),constrained_layout=True)
#plt.tight_layout(pad=5.4, w_pad=5.5, h_pad=5.4)
tick_params(direction = 'in', width=2, labelsize=20)
ax.set_ylabel("N", fontsize = '30')   
ax.set_xlabel(r"$\rm Wavelength \ (\mu m)$", fontsize = '30')
# plt.xlim(0,4)
ax.plot(lam, abs(N1), 'o', markeredgecolor = "black", markerfacecolor = 'red', label = "Absolute value of complex index N")#, label = "Digitized Data")
ax.plot(lam, func_Re(lam, *popt_Re), color = "black", linewidth = 3, label = r"$\rm \gamma_{fit} = %.2f$ " %popt_Re[0])# "\n" r"J = %f" %tuple(popt))
ax.legend(loc='upper center', fontsize='12')    
# plot(lam, Tm,  label=r'$\rm T_{nk: Ag}$', color='green')
# plot(lam, Rm + Tm,  label=r'$\rm R+T_{nk: Ag}$', color='blue')

plt.setp(ax.spines.values(), linewidth=2)


# tick_params(direction = 'in', width=2, labelsize=20)
# ax1.set_ylabel("k", fontsize = '30')   
# ax1.set_xlabel(r"$\rm Wavelength \ (\mu m)$", fontsize = '30')
# # plt.xlim(0,4)
# ax1.plot(lam, k, 'o', markeredgecolor = "black", markerfacecolor = 'red', label = "Imaginary index k")#, label = "Digitized Data")
# ax.plot(lam, func_Im(lam, *popt_Im), color = "black", linewidth = 3, label = r"$\rm \gamma_{fit} = %.2f$ " %popt_Re[0])# "\n" r"J = %f" %tuple(popt))
# # ax.plot(lam, func(lam, *popt), color = "black", linewidth = 3, label = r"$\rm \gamma_{fit} = %.2f$ " %popt[0])# "\n" r"J = %f" %tuple(popt))
    
# # plot(lam, Tm,  label=r'$\rm T_{nk: Ag}$', color='green')
# # plot(lam, Rm + Tm,  label=r'$\rm R+T_{nk: Ag}$', color='blue')

# plt.setp(ax1.spines.values(), linewidth=2)

# plt.tick_params(left = False, bottom = False)   

# ax1.legend(loc='center right', fontsize='12')
# ax.axvline(x = c0/(wp*1e-6), color = 'black', linewidth = 2)
plt.savefig("AgPalik_OurRange_Abs_IndexBothFit.pdf")
plt.savefig("AgPalik_OurRange_Abs_IndexBothFit.png")

# plt.show()
