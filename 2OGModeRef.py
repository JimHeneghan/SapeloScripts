import numpy as np
from scipy import *
from pylab import *
import math
import cmath
from ctypes import *
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['agg.path.chunksize'] = 10000
#all units are in m^-1
wp = 1.15136316e16
gamma = 9.79125662e13
d = 50e-9
imp0 = 376.730313
w = linspace (100e12, 3600e12,  100)
# k0 = k*1e12
c0 = 3e8
eps1 = 1 + (wp*wp)/(w*(1j*gamma-w))
dx = 5e-8
dy = 5e-8
n1 = np.sqrt(eps1)

#using the equations in chapter 2.2.r of Macleod
#assuming the impedance of free space cancels out
#assuming the incident media is vacuum with k0 = 0
# 
# unlabled equation on p 38 in Macleod after eqn 2.88 
delta1 = n1*d*(w/c0)#*2*math.pi


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



Nx = 41
Ny = 41
NFREQs = 1000
nref = 1.0
ntra = 3.42

freq = np.zeros(NFREQs, dtype = np.double)
m  = np.zeros(Nx, dtype = np.int)
n  = np.zeros(Ny, dtype = np.int)

kx = np.zeros(Nx, dtype = np.double)
ky = np.zeros(Ny, dtype = np.double)

Sxr    = np.zeros((Nx, Ny), dtype = np.complex)
Syr    = np.zeros((Nx, Ny), dtype = np.complex)

kzR    = np.zeros((Nx, Ny), dtype = np.complex)
kzT    = np.zeros((Nx, Ny), dtype = np.complex)

Esr    = np.zeros((Nx, Ny), dtype = np.complex)

# SR1    = np.zeros((Nx, Ny), dtype = np.complex)

# ExRefI = np.zeros((Nx, Ny), dtype = np.double)
# ExRefR = np.zeros((Nx, Ny), dtype = np.double)
# ExRef  = np.zeros((Nx, Ny), dtype = np.complex)

# EyRefI = np.zeros((Nx, Ny), dtype = np.double)
# EyRefR = np.zeros((Nx, Ny), dtype = np.double)
# EyRef  = np.zeros((Nx, Ny), dtype = np.complex)

# EzRefI = np.zeros((Nx, Ny), dtype = np.double)
# EzRefR = np.zeros((Nx, Ny), dtype = np.double)
# EzRef  = np.zeros((Nx, Ny), dtype = np.complex)

# ExTra  = np.zeros((Nx, Ny), dtype = np.complex)
# EyTra  = np.zeros((Nx, Ny), dtype = np.complex)
# EzTra  = np.zeros((Nx, Ny), dtype = np.complex)



# ExIncR = np.zeros(NFREQs, dtype = np.double)
# ExIncI = np.zeros(NFREQs, dtype = np.double)
# ExInc  = np.zeros((Nx, Ny), dtype = np.complex)

# EyIncR = np.zeros(NFREQs, dtype = np.double)
# EyIncI = np.zeros(NFREQs, dtype = np.double)
# EyInc  = np.zeros((Nx, Ny), dtype = np.complex)

# EzIncR = np.zeros(NFREQs, dtype = np.double)
# EzIncI = np.zeros(NFREQs, dtype = np.double)
# EzInc  = np.zeros((Nx, Ny), dtype = np.complex)

# Esr  = np.zeros((Nx, Ny), dtype = np.complex)


ref   = np.zeros((NFREQs, Nx, Ny), dtype = np.double)
REF   = np.zeros(NFREQs, dtype = np.double)

tra   = np.zeros((NFREQs, Nx, Ny), dtype = np.double)
TRA   = np.zeros(NFREQs, dtype = np.double)
for i in range (0, Nx):
	m[i] = i - (Nx  - 1)/2	
print(m)
for i in range (0, Ny):
	n[i] = i - (Ny  - 1)/2

freq   = np.loadtxt("freq.txt",  usecols=(0), skiprows= 1, unpack =True)

ExR    = np.loadtxt("ExRef.txt", usecols=(0), skiprows= 1, unpack =True)
ExI    = np.loadtxt("ExRef.txt", usecols=(1), skiprows= 1, unpack =True)

EyR    = np.loadtxt("EyRef.txt", usecols=(0), skiprows= 1, unpack =True)
EyI    = np.loadtxt("EyRef.txt", usecols=(1), skiprows= 1, unpack =True)

EzR    = np.loadtxt("EzRef.txt", usecols=(0), skiprows= 1, unpack =True)
EzI    = np.loadtxt("EzRef.txt", usecols=(1), skiprows= 1, unpack =True)

ExTR   = np.loadtxt("ExTra.txt", usecols=(0), skiprows= 1, unpack =True)
ExTI   = np.loadtxt("ExTra.txt", usecols=(1), skiprows= 1, unpack =True)

EyTR   = np.loadtxt("EyTra.txt", usecols=(0), skiprows= 1, unpack =True)
EyTI   = np.loadtxt("EyTra.txt", usecols=(1), skiprows= 1, unpack =True)

EzTR   = np.loadtxt("EzTra.txt", usecols=(0), skiprows= 1, unpack =True)
EzTI   = np.loadtxt("EzTra.txt", usecols=(1), skiprows= 1, unpack =True)

ExIncR = np.loadtxt("ExInc1.txt", usecols=(0), skiprows= 1, unpack =True)
ExIncI = np.loadtxt("ExInc1.txt", usecols=(1), skiprows= 1, unpack =True)

EyIncR = np.loadtxt("EyInc1.txt", usecols=(0), skiprows= 1, unpack =True)
EyIncI = np.loadtxt("EyInc1.txt", usecols=(1), skiprows= 1, unpack =True)

EzIncR = np.loadtxt("EzInc1.txt", usecols=(0), skiprows= 1, unpack =True)
EzIncI = np.loadtxt("EzInc1.txt", usecols=(1), skiprows= 1 , unpack =True)

ExRef = ExR + 1j*ExI
EyRef = EyR + 1j*EyI
EzRef = EzR + 1j*EzI

ExTra = ExTR + 1j*ExTI
EyTra = EyTR + 1j*EyTI
EzTra = EzTR + 1j*EzTI

ExInc = ExIncR + 1j*ExIncI
EyInc = EyIncR + 1j*EyIncI
EzInc = EzIncR + 1j*EzIncI

ExRef = np.reshape(ExRef, (NFREQs, Nx, Ny), order='C')
EyRef = np.reshape(EyRef, (NFREQs, Nx, Ny), order='C')
EzRef = np.reshape(EzRef, (NFREQs, Nx, Ny), order='C')

ExTra = np.reshape(ExTra, (NFREQs, Nx, Ny), order='C')
EyTra = np.reshape(EyTra, (NFREQs, Nx, Ny), order='C')
EzTra = np.reshape(EzTra, (NFREQs, Nx, Ny), order='C')

ExInc = np.reshape(ExInc, (NFREQs, Nx, Ny), order='C')
EyInc = np.reshape(EyInc, (NFREQs, Nx, Ny), order='C')
EzInc = np.reshape(EzInc, (NFREQs, Nx, Ny), order='C')

kx = -2*math.pi*m/(Nx*dx)
ky = -2*math.pi*n/(Ny*dy)

print(len(EzInc))
for ff in range (0, NFREQs):

	Esr = 0
	lam = c0/freq[ff]
	# print(lam)
	k0 = 2*math.pi/lam
	kzinc = k0*nref

	for i in range (0, Nx):
		for j in range (0, Ny):
			kzR[i, j] = cmath.sqrt((k0*nref)**2 - kx[i]**2 - ky[j]**2)
			kzT[i, j] = cmath.sqrt((k0*ntra)**2 - kx[i]**2 - ky[j]**2)
	Esr = np.sqrt(abs(ExInc[ff])**2 + abs(EyInc[ff])**2 + abs(EzInc[ff])**2)#ExInc[ff]*conj(ExInc[ff]) + EyInc[ff]*conj(EyInc[ff]) + EzInc[ff]*conj(EzInc[ff]))
	
	Sxr = ExRef[ff]/Esr
	Syr = EyRef[ff]/Esr
	Szr = EzRef[ff]/Esr

	Sxt = ExTra[ff]/Esr
	Syt = EyTra[ff]/Esr
	Szt = EzTra[ff]/Esr

	# print(Sxr)
	
	Sxrfft = fftshift(fft2(Sxr))/(Nx*Ny)
	Syrfft = fftshift(fft2(Syr))/(Nx*Ny)
	Szrfft = fftshift(fft2(Szr))/(Nx*Ny)

	Sxtfft = fftshift(fft2(Sxt))/(Nx*Ny)
	Sytfft = fftshift(fft2(Syt))/(Nx*Ny)
	Sztfft = fftshift(fft2(Szt))/(Nx*Ny)

	Sref = abs(Sxrfft)**2 + abs(Syrfft)**2 + abs(Szrfft)**2 #conj(Sxrfft)*Sxrfft + conj(Syrfft)*Syrfft + conj(Szrfft)*Szrfft
	Stra = abs(Sxtfft)**2 + abs(Sytfft)**2 + abs(Sztfft)**2 #conj(Sxtfft)*Sxrfft + conj(Sytfft)*Syrfft + conj(Sztfft)*Szrfft

	ref[ff] = Sref*(real(kzR/kzinc))
	tra[ff] = Stra*(real(kzT/kzinc))

for ff in range(0, NFREQs):
	for j in range(0, Ny):
		for i in range(0, Nx):
			REF[ff] =  REF[ff] + ref[ff, i, j]
			TRA[ff] =  TRA[ff] + tra[ff, i, j]

fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(direction = 'in', width=2, labelsize=20)
plt.ylabel("R", fontsize = '30')   
plt.xlabel(r"$\rm Wavelength \ (\mu m)$", fontsize = '30')
plt.xlim(0,10)
plt.ylim(0,1)

plt.plot((c0/freq)*1e6, REF, label = r'$\rm R_{FDTD \ Ag Pattern \ Toy \ nm^{3}}$', color = "red", linewidth = 6)
plt.plot((c0/freq)*1e6, TRA, label = r'$\rm T_{FDTD \ Ag Pattern \ Toy \ nm^{3}}$', color = "black", linewidth = 4)
plt.plot((c0/freq)*1e6, (1-(REF+TRA)), label = r'$\rm A_{FDTD \ Ag Pattern \ Toy \ nm^{3}}$', color = "limegreen", linewidth = 2)

# plt.plot((c0*2*math.pi/(w))*1e6, Rm, label=r'$\rm R_{TM}$', color='yellow', linewidth = 2))

plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(left = False, bottom = False)   
ax.legend(loc='upper right', fontsize='22')
ax.axhline(y =1, color = 'black')
ax.axvline(x =0.1, color = 'black')
plt.savefig("ToyJimStructure.pdf")
plt.savefig("ToyJimStructure.png")

# plt.show()
