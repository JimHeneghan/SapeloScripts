##############################################################################
# This script loads the n, k data from Palik and fits either:
#     (i)   a Drude model to the data (use_debye = OFF),
#     (ii)  a Drude model plus a Debye model to the data (use_debye = ON).
# There are three was to fit to the data:
#     (i)   fit to the complex data (fit_type = BOTH),
#     (ii)  fit to the real data (fit_type = REALONlY),  
#     (ii)  fit to the imaginary data (fit_type = IMAGONLY).
# The range of which the data is fitted can be changed by 
# changing the values of START and STOP.
# The calculations are performed in eV, these values are converted to rad/s. 
##############################################################################

from pylab import *
from scipy.optimize import curve_fit

# Physical constants
hbar = 1.0545718E-34
qe = 1.602176634E-19

# Load data from Palik
um, k, n = loadtxt('AgPalikLum.txt', usecols=(0,2,3), skiprows = 1, unpack=True)
# Calculate microns from eV
um = um/100
ev = 1.23984193/um

# Constants
ON=1
OFF=0
START=0
STOP=len(ev)
REALONLY=1e10
IMAGONLY=1/REALONLY
BOTH=1

# Drude model from Sullivan page 28 
def drude(w, wp, nuc):
    eps = 1.0 + wp**2/(w*(1j*nuc-w))
    return eps

# Debye Model from Sullivan page 24
def debye(w, chi1, t0):
    eps = chi1/(1+1j*w*t0)
    return eps

# Wrapper function required for fitting to complex data
def wrapdrude(ev, wp2, nuc):
    myeps=1j*zeros(ev.size)
    res=zeros(ev.size)
    for i in range(0,ev.size):
        myeps[i] = drude(ev[i], wp2, nuc)
        res[i] = real(sqrt(myeps[i]))
    for i in range(ev.size//2,ev.size):
        myeps[i] = drude(ev[i], wp2, nuc)
        res[i] = -imag(sqrt(myeps[i]))
    return res
    
def wrapboth(ev, wp2, nuc, chi1, t0):
    myeps=1j*zeros(ev.size)
    res=zeros(ev.size)
    for i in range(0,ev.size):
        myeps[i] = drude(ev[i], wp2, nuc) + debye(ev[i], chi1, t0)
        res[i] = real(sqrt(myeps[i]))
    for i in range(ev.size//2,ev.size):
        myeps[i] = drude(ev[i], wp2, nuc) + debye(ev[i], chi1, t0)
        res[i] = -imag(sqrt(myeps[i]))
    return res

# Initial Parameter Guesses
wp = 50.0
nuc = 0.1
chi1 = 50.0
t0 = 0.01

# Specify type of fit
use_debye = OFF
fit_type = BOTH

# Restrict range of fit
ev = ev[START:STOP]
um = um[START:STOP]
n = n[START:STOP]
k = k[START:STOP]

# Construct packed arrays for wrap functions
x=concatenate([ev,ev])
y=concatenate([n,k])

# Use the packed sigma values to select fit type 
sig = ones(len(ev))
s=concatenate([sig, fit_type*sig])

# Fit the data, convert parameters from ev to rad/s etc. and print
if (use_debye==OFF):
    params=[wp, nuc]
    res  = curve_fit(wrapdrude, x, y, params, s)
    wp = res[0][0]
    nuc = res[0][1]
    myeps = drude(ev, wp, nuc)
    print(wp)
    print(f'omega_p = {qe*wp/hbar:15.8e} rad/s')
    print(f'nu_c = {qe*nuc/hbar:15.8e} rad/s')
else:
    params=[wp, nuc, chi1, t0]
    res = curve_fit(wrapboth, x, y, params, s)
    wp = res[0][0]
    nuc = res[0][1]
    chi1 = res[0][2]
    t0 = res[0][3]
    myeps = drude(ev, wp, nuc) + debye(ev, chi1, t0)
    print(wp)
    print(f'omega_p = {qe*wp/hbar:15.8e} rad/s')
    print(f'nu_c = {qe*nuc/hbar:15.8e} rad/s')
    print(f'chi_1 = {chi1:15.8e} (unitless)')
    print(f't_0 = {hbar*t0/(2*pi*qe):15.8e} s')


# Calculate n and k for plotting
myn = real(sqrt(myeps))
myk = -imag(sqrt(myeps))


# Plot data and fits
xscale('log')
yscale('log')
plot(um, myn, linewidth=2, c='b', label="Fit to n")
plot(um, myk, linewidth=2, c='r', label="Fit to k")
scatter(um, n, s=20, c='b', label="n from Palik")	
scatter(um, k, s=20, c='r', label="k from Palik")
plt.xlabel(r'Wavelength ($\mathrm{\mu}$m)')
plt.ylabel(r'n, k')	
legend()	
show()


