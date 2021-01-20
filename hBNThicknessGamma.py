import numpy as np
import math
import collections as cl
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from scipy import *
from scipy.optimize import curve_fit
from pylab import *

thick  = [60, 105, 210, 490, 1020, 1990, 6400]
Newd   = [50, 60, 70, 80, 90, 100]
shades = ['red', 'orange', 'gold', 'green', 'blue', 'purple', 'magenta']
cols   = ['red', 'darkgoldenrod', 'green', 'blue', 'purple']
gamma  = [40.047, 27.2056, 9.622, 17.4781, 8.7441, 9.0982, 1.3777]
fig, (ax1, ax3) = plt.subplots(2, figsize = (18,30))

plt.setp(ax1.spines.values(), linewidth=2)
ax1.tick_params(direction = 'in', width=2, labelsize=30)
ax1.tick_params(axis = 'x', direction = 'in', width=2, labelsize=0)
ax1.set_xlim(1200,1700)
plt.rc('axes', linewidth=2) 

q = 0
m = 0
for z in range (0,7):   
    F1 = "CaldwellDigi/hBN%dnmRef.txt" %thick[z]
    k_data, R_exp = loadtxt(F1, usecols=(0,1,),  unpack = True)
    eps_inf = 4.87
    s = 1.83
    omega_nu = 1372
    d = thick[z]*1e-7
    imp0 = 376.730313
    chi2 = 20
    inc = math.pi*25.0/180.0
    
    nb = np.zeros(len(k_data), dtype=np.complex64)
    #gamma = linspace (1, 300, 100001)
    def func(k, gamma, J):
        lam = (1/(k*100))*1e6
        nb = np.sqrt(1+ 0.33973 + ((0.81070 * (lam**2))/(lam**2 - 0.10065**2))
                         + ((0.19652*(lam**2))/(lam**2 - 29.87**2)) + ((4.52469*(lam**2))/(lam**2 - 53.82**2)))
        eps1 = eps_inf + (s*(omega_nu**2))/((omega_nu**2) + 1j*gamma*k - k*k)
        n1 = np.sqrt(eps1)
        delta1 = n1*d*k*2*math.pi*cos(inc)
        
        eta0 = imp0
        eta1 = (n1)*imp0
        eta2 = imp0*nb
        
        Y =  (eta2*cos(delta1) + 1j*eta1*sin(delta1))/(cos(delta1) + 1j*(eta2/eta1)*sin(delta1))

        
        return abs(((eta0 - Y)/(eta0 + Y))*conj((eta0 - Y)/(eta0 + Y))*J)

    popt, pcov = curve_fit(func, k_data, R_exp, bounds = (0, [300.0, 1.0]))
    print(popt)
    

    #plt.plot(k, R_exp, label = "grabbed data %d nm" %thick[z])
    #plt.plot(k, Rlow, label = r"fitted gamma is %g $ \rm cm^{-1}$ ""\n $\chi^{2} = %g$ \n $\\theta = %g^{\circ}$ \n hBN is %d nm thick" %(gammaLow, chilow, inc*180/math.pi, thick[z]))
    ax1.plot(k_data, func(k_data, *popt), linewidth = 6, color = shades[z], label = r"$d\rm_{hBN}  \ = \ %d \ nm$" %thick[z])# label = r"$\rm \gamma_{fit} = %.2f \ J = %.2f $ " %(popt[0], popt[1]))# "\n" r"J = %f" %tuple(popt))
    print("thickness = %d \t gamma = %0.2f \t J = %0.2f " %(thick[z], popt[0], popt[1]))
    ax1.plot(k_data, R_exp, 'o',  markersize=10, markeredgecolor = "black", linewidth = 5, markerfacecolor = shades[z])#, label = "Digitized Data")#  markersize=3, markeredgecolor = "black", markerface

        

    #    legend(loc='lower center', fontsize='30')
ax1.set_ylabel(r"$R$", fontsize = '70')
ax1.set_xlabel(r'$\rm Frequency \ (cm^{-1})$', fontsize = '70')

# ax1.set_xlabel(r'$\rm Wavenumber \ (cm^{-1})$', fontsize = '70')
ax1.legend(loc='center right', fontsize='30')
ax1.axvline(x = 1370, linestyle = "dashed", color = 'black', linewidth = 2)
ax1.axvline(x = 1610, linestyle = "dashed", color = 'black', linewidth = 2)

ax3.tick_params(direction = 'in', width=2, labelsize=30, size = 4)
ax3.tick_params(which ='minor', direction = 'in', width=2, size = 2)
ax3.set_ylabel(r'$\gamma_{x,y} \ \rm  (rad \ s^{-1})$', fontsize = '70')
ax3.set_xlabel(r'$ d\rm_{hBN}$', fontsize = '70')

# thick = [60e-9, 105e-9, 210e-9, 490e-9, 1020e-9, 1990e-9, 6400e-9]

def func(d, m, c):
    return m*d + c

popt, pcov = curve_fit(func, log(np.asarray(thick)) , log(np.asarray(gamma)))
c0 = 3e8

x = np.asarray(thick)
y = np.asarray(gamma) 
y = y*100
logx = np.log(x)
logy = np.log(y)
coeffs = np.polyfit(logx,logy,deg=1)
poly = np.poly1d(coeffs)
yfit = lambda x: np.exp(poly(np.log(x)))
 
num1 = yfit(83)
# num2 = yfit(1000)

num1 = 2*math.pi*c0*(exp(num1)*100/1e12)
print("yfit for 83 nm is %5.3f" %num1)
print(r"$\rm log(\gamma) = %5.3f log(d) \ + \ log(%5.3f)$"  %tuple(popt))


ax3.loglog(thick, 2*math.pi*c0*y, "ro", markersize=25, markeredgecolor = "black", markeredgewidth = 5, markerfacecolor = 'red', label = r"$\mathrm{Calculated \ \mathit{\gamma}_{x,y}}$")# shades[z])#, label = r"$\rm Calculated \ \gamma: \ d = %d $" %thick[z], zorder = 5)
ax3.loglog(x, 2*math.pi*c0*yfit(x), color = "black", linewidth=6, label = r"$\mathrm{Power-law \ fit \ to} \ \gamma_{x,y} \mathrm{ \ vs \ } d\rm_{hBN} \mathrm{\ data}$", zorder = 1)
for i in range(0, 6):
    ax3.scatter(Newd[i], 2*math.pi*c0*yfit(Newd[i]), s=600, facecolors = "none", edgecolor = shades[i],linewidth=8.0, zorder = 6, 
        label = r"$\gamma_{x,y} \mathrm{ \ (%d \ nm) = %0.2f \times 10^{12} \ (rad \ s^{-1})} $" %(Newd[i],2*math.pi*c0*yfit(Newd[i])/1e12))


plt.gcf().subplots_adjust(bottom=0.24, top = 0.99)
leg = ax3.legend(fontsize='30', framealpha=0.2, loc = 'lower center', fancybox = False, 
 bbox_to_anchor=(-0.16, -0.45, 1.29, .175),mode="expand", borderaxespad=0., ncol = 2)

leg.get_frame().set_edgecolor('black')

leg.get_frame().set_linewidth(4)

plt.ylim(4.5e12,8.5e12)
ax3.set_xlim(47, 125)
ax3.axvline(x =80, color = 'black')
ax3.axhline(y = 2*math.pi*c0*(exp(3.484986312)*100), color = 'black')
plt.setp(ax3.spines.values(), linewidth=2)

plt.savefig("GammaFind.png")
plt.savefig("GammaFind.pdf")
# plt.show()
