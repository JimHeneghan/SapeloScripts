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
##############################################################################
wp = 1.15136316e16
gamma = 9.79125662e13
eps_inf = 4.87
s = 1.83
omega_nu = 137200
gamma = 3198.7#/(2*math.pi)#700.01702
d = 80e-9
imp0 = 376.730313
k00 = linspace (100000, 2000000, 20000)
c0 = 3e8
eps1 = eps_inf + (s*(omega_nu**2))/((omega_nu**2) + 1j*gamma*k00 - k00*k00)

n1 = np.sqrt(eps1)

#using the equations in chapter 2.2.r of Macleod
#assuming the impedance of free space cancels out
#assuming the incident media is vacuum with k00 = 0

# unlabled equation on p 38 in Macleod after eqn 2.88 
delta1 = n1*d*k00*2*math.pi


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
##############################################################################
##############################################################################

dx     = 20e-9
dy     = 20e-9
nref   = 1.0
ntra   = 3.42
NFREQs = 500


###########################################################################
###########################  Pitch = 2.24 um ##############################
###########################################################################
Nx     = 113
Ny     = 193

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

EyIncR = np.zeros((NFREQs, Nx, Ny), dtype = np.complex) 
EyIncI = np.zeros((NFREQs, Nx, Ny), dtype = np.complex)
EzIncR = np.zeros((NFREQs, Nx, Ny), dtype = np.complex)
EzIncI = np.zeros((NFREQs, Nx, Ny), dtype = np.complex)


ref   = np.zeros((NFREQs, Nx, Ny), dtype = np.double)
REFOur   = np.zeros(NFREQs, dtype = np.double)

tra   = np.zeros((NFREQs, Nx, Ny), dtype = np.double)
TRAOur   = np.zeros(NFREQs, dtype = np.double)
for i in range (0, Nx):
	m[i] = i - (Nx  - 1)/2	
print(m)
for i in range (0, Ny):
	n[i] = i - (Ny  - 1)/2

freq   = np.loadtxt("2_24/Ref/freq.txt",  usecols=(0), skiprows= 1, unpack =True)

ExR    = np.loadtxt("2_24/Ref/ExRef.txt", usecols=(0), skiprows= 1, unpack =True)
ExI    = np.loadtxt("2_24/Ref/ExRef.txt", usecols=(1), skiprows= 1, unpack =True)

EyR    = np.loadtxt("2_24/Ref/EyRef.txt", usecols=(0), skiprows= 1, unpack =True)
EyI    = np.loadtxt("2_24/Ref/EyRef.txt", usecols=(1), skiprows= 1, unpack =True)

EzR    = np.loadtxt("2_24/Ref/EzRef.txt", usecols=(0), skiprows= 1, unpack =True)
EzI    = np.loadtxt("2_24/Ref/EzRef.txt", usecols=(1), skiprows= 1, unpack =True)

ExTR   = np.loadtxt("2_24/Ref/ExTra.txt", usecols=(0), skiprows= 1, unpack =True)
ExTI   = np.loadtxt("2_24/Ref/ExTra.txt", usecols=(1), skiprows= 1, unpack =True)

EyTR   = np.loadtxt("2_24/Ref/EyTra.txt", usecols=(0), skiprows= 1, unpack =True)
EyTI   = np.loadtxt("2_24/Ref/EyTra.txt", usecols=(1), skiprows= 1, unpack =True)

EzTR   = np.loadtxt("2_24/Ref/EzTra.txt", usecols=(0), skiprows= 1, unpack =True)
EzTI   = np.loadtxt("2_24/Ref/EzTra.txt", usecols=(1), skiprows= 1, unpack =True)

ExIncR = np.loadtxt("2_24/Ref/ExInc.txt", usecols=(0), skiprows= 1, unpack =True)
ExIncI = np.loadtxt("2_24/Ref/ExInc.txt", usecols=(1), skiprows= 1, unpack =True)

EyIncR = np.loadtxt("2_24/Ref/EyInc.txt", usecols=(0), skiprows= 1, unpack =True)
EyIncI = np.loadtxt("2_24/Ref/EyInc.txt", usecols=(1), skiprows= 1, unpack =True)

EzIncR = np.loadtxt("2_24/Ref/EzInc.txt", usecols=(0), skiprows= 1, unpack =True)
EzIncI = np.loadtxt("2_24/Ref/EzInc.txt", usecols=(1), skiprows= 1 , unpack =True)

ExRef = ExR + 1j*ExI
EyRef = EyR + 1j*EyI
EzRef = EzR + 1j*EzI

ExTra = ExTR + 1j*ExTI
EyTra = EyTR + 1j*EyTI
EzTra = EzTR + 1j*EzTI

ExInc = (ExIncR + 1j*ExIncI)
EyInc = (EzIncR + 1j*EzIncI)#(EyIncR + 1j*EyIncI)
EzInc = (EzIncR + 1j*EzIncI)

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
			REFOur[ff] =  REFOur[ff] + ref[ff, i, j]
			TRAOur[ff] =  TRAOur[ff] + tra[ff, i, j]
###########################################################################
###########################  Pitch = 2.281 ################################
###########################################################################

Nx     = 113
Ny     = 193
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

EyIncR = np.zeros((NFREQs, Nx, Ny), dtype = np.complex) 
EyIncI = np.zeros((NFREQs, Nx, Ny), dtype = np.complex)
EzIncR = np.zeros((NFREQs, Nx, Ny), dtype = np.complex)
EzIncI = np.zeros((NFREQs, Nx, Ny), dtype = np.complex)


ref   = np.zeros((NFREQs, Nx, Ny), dtype = np.double)
REFp2_281   = np.zeros(NFREQs, dtype = np.double)

tra   = np.zeros((NFREQs, Nx, Ny), dtype = np.double)
TRAp2_281   = np.zeros(NFREQs, dtype = np.double)
for i in range (0, Nx):
	m[i] = i - (Nx  - 1)/2	
print(m)
for i in range (0, Ny):
	n[i] = i - (Ny  - 1)/2

freq   = np.loadtxt("2_281um/Ref/freq.txt",  usecols=(0), skiprows= 1, unpack =True)

ExR    = np.loadtxt("2_281um/Ref/ExRef.txt", usecols=(0), skiprows= 1, unpack =True)
ExI    = np.loadtxt("2_281um/Ref/ExRef.txt", usecols=(1), skiprows= 1, unpack =True)

EyR    = np.loadtxt("2_281um/Ref/EyRef.txt", usecols=(0), skiprows= 1, unpack =True)
EyI    = np.loadtxt("2_281um/Ref/EyRef.txt", usecols=(1), skiprows= 1, unpack =True)

EzR    = np.loadtxt("2_281um/Ref/EzRef.txt", usecols=(0), skiprows= 1, unpack =True)
EzI    = np.loadtxt("2_281um/Ref/EzRef.txt", usecols=(1), skiprows= 1, unpack =True)

ExTR   = np.loadtxt("2_281um/Ref/ExTra.txt", usecols=(0), skiprows= 1, unpack =True)
ExTI   = np.loadtxt("2_281um/Ref/ExTra.txt", usecols=(1), skiprows= 1, unpack =True)

EyTR   = np.loadtxt("2_281um/Ref/EyTra.txt", usecols=(0), skiprows= 1, unpack =True)
EyTI   = np.loadtxt("2_281um/Ref/EyTra.txt", usecols=(1), skiprows= 1, unpack =True)

EzTR   = np.loadtxt("2_281um/Ref/EzTra.txt", usecols=(0), skiprows= 1, unpack =True)
EzTI   = np.loadtxt("2_281um/Ref/EzTra.txt", usecols=(1), skiprows= 1, unpack =True)

ExIncR = np.loadtxt("2_281um/Ref/ExInc.txt", usecols=(0), skiprows= 1, unpack =True)
ExIncI = np.loadtxt("2_281um/Ref/ExInc.txt", usecols=(1), skiprows= 1, unpack =True)

EyIncR = np.loadtxt("2_281um/Ref/EyInc.txt", usecols=(0), skiprows= 1, unpack =True)
EyIncI = np.loadtxt("2_281um/Ref/EyInc.txt", usecols=(1), skiprows= 1, unpack =True)

EzIncR = np.loadtxt("2_281um/Ref/EzInc.txt", usecols=(0), skiprows= 1, unpack =True)
EzIncI = np.loadtxt("2_281um/Ref/EzInc.txt", usecols=(1), skiprows= 1 , unpack =True)

ExRef = ExR + 1j*ExI
EyRef = EyR + 1j*EyI
EzRef = EzR + 1j*EzI

ExTra = ExTR + 1j*ExTI
EyTra = EyTR + 1j*EyTI
EzTra = EzTR + 1j*EzTI

ExInc = (ExIncR + 1j*ExIncI)
EyInc = (EyIncR + 1j*EyIncI)
EzInc = (EzIncR + 1j*EzIncI)

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
			REFp2_281[ff] =  REFp2_281[ff] + ref[ff, i, j]
			TRAp2_281[ff] =  TRAp2_281[ff] + tra[ff, i, j]

###########################################################################
###########################  Pitch = 2.36 um ##############################
###########################################################################

Nx     = 119
Ny     = 205

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

EyIncR = np.zeros((NFREQs, Nx, Ny), dtype = np.complex) 
EyIncI = np.zeros((NFREQs, Nx, Ny), dtype = np.complex)
EzIncR = np.zeros((NFREQs, Nx, Ny), dtype = np.complex)
EzIncI = np.zeros((NFREQs, Nx, Ny), dtype = np.complex)


ref   = np.zeros((NFREQs, Nx, Ny), dtype = np.double)
REFPaperAg   = np.zeros(NFREQs, dtype = np.double)

tra   = np.zeros((NFREQs, Nx, Ny), dtype = np.double)
TRAPaperAg   = np.zeros(NFREQs, dtype = np.double)
for i in range (0, Nx):
	m[i] = i - (Nx  - 1)/2	
print(m)
for i in range (0, Ny):
	n[i] = i - (Ny  - 1)/2

freq   = np.loadtxt("2_36um/Ref/freq.txt",  usecols=(0), skiprows= 1, unpack =True)

ExR    = np.loadtxt("2_36um/Ref/ExRef.txt", usecols=(0), skiprows= 1, unpack =True)
ExI    = np.loadtxt("2_36um/Ref/ExRef.txt", usecols=(1), skiprows= 1, unpack =True)

EyR    = np.loadtxt("2_36um/Ref/EyRef.txt", usecols=(0), skiprows= 1, unpack =True)
EyI    = np.loadtxt("2_36um/Ref/EyRef.txt", usecols=(1), skiprows= 1, unpack =True)

EzR    = np.loadtxt("2_36um/Ref/EzRef.txt", usecols=(0), skiprows= 1, unpack =True)
EzI    = np.loadtxt("2_36um/Ref/EzRef.txt", usecols=(1), skiprows= 1, unpack =True)

ExTR   = np.loadtxt("2_36um/Ref/ExTra.txt", usecols=(0), skiprows= 1, unpack =True)
ExTI   = np.loadtxt("2_36um/Ref/ExTra.txt", usecols=(1), skiprows= 1, unpack =True)

EyTR   = np.loadtxt("2_36um/Ref/EyTra.txt", usecols=(0), skiprows= 1, unpack =True)
EyTI   = np.loadtxt("2_36um/Ref/EyTra.txt", usecols=(1), skiprows= 1, unpack =True)

EzTR   = np.loadtxt("2_36um/Ref/EzTra.txt", usecols=(0), skiprows= 1, unpack =True)
EzTI   = np.loadtxt("2_36um/Ref/EzTra.txt", usecols=(1), skiprows= 1, unpack =True)

ExIncR = np.loadtxt("2_36um/Ref/ExInc.txt", usecols=(0), skiprows= 1, unpack =True)
ExIncI = np.loadtxt("2_36um/Ref/ExInc.txt", usecols=(1), skiprows= 1, unpack =True)

EyIncR = np.loadtxt("2_36um/Ref/EyInc.txt", usecols=(0), skiprows= 1, unpack =True)
EyIncI = np.loadtxt("2_36um/Ref/EyInc.txt", usecols=(1), skiprows= 1, unpack =True)

EzIncR = np.loadtxt("2_36um/Ref/EzInc.txt", usecols=(0), skiprows= 1, unpack =True)
EzIncI = np.loadtxt("2_36um/Ref/EzInc.txt", usecols=(1), skiprows= 1 , unpack =True)

ExRef = ExR + 1j*ExI
EyRef = EyR + 1j*EyI
EzRef = EzR + 1j*EzI

ExTra = ExTR + 1j*ExTI
EyTra = EyTR + 1j*EyTI
EzTra = EzTR + 1j*EzTI

ExInc = (ExIncR + 1j*ExIncI)
EyInc = (EzIncR + 1j*EzIncI)#(EyIncR + 1j*EyIncI)
EzInc = (EzIncR + 1j*EzIncI)

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
			REFPaperAg[ff] =  REFPaperAg[ff] + ref[ff, i, j]
			TRAPaperAg[ff] =  TRAPaperAg[ff] + tra[ff, i, j]



# ###########################################################################
# ###########################  Pitch = 2.42 um ##############################
# ###########################################################################

Nx     = 125
Ny     = 215
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

EyIncR = np.zeros((NFREQs, Nx, Ny), dtype = np.complex) 
EyIncI = np.zeros((NFREQs, Nx, Ny), dtype = np.complex)
EzIncR = np.zeros((NFREQs, Nx, Ny), dtype = np.complex)
EzIncI = np.zeros((NFREQs, Nx, Ny), dtype = np.complex)


ref   = np.zeros((NFREQs, Nx, Ny), dtype = np.double)
REF2_42   = np.zeros(NFREQs, dtype = np.double)

tra   = np.zeros((NFREQs, Nx, Ny), dtype = np.double)
TRA2_42   = np.zeros(NFREQs, dtype = np.double)
for i in range (0, Nx):
	m[i] = i - (Nx  - 1)/2	
print(m)
for i in range (0, Ny):
	n[i] = i - (Ny  - 1)/2

freq   = np.loadtxt("2_42_48um/Ref/freq.txt",  usecols=(0), skiprows= 1, unpack =True)

ExR    = np.loadtxt("2_42_48um/Ref/ExRef.txt", usecols=(0), skiprows= 1, unpack =True)
ExI    = np.loadtxt("2_42_48um/Ref/ExRef.txt", usecols=(1), skiprows= 1, unpack =True)

EyR    = np.loadtxt("2_42_48um/Ref/EyRef.txt", usecols=(0), skiprows= 1, unpack =True)
EyI    = np.loadtxt("2_42_48um/Ref/EyRef.txt", usecols=(1), skiprows= 1, unpack =True)

EzR    = np.loadtxt("2_42_48um/Ref/EzRef.txt", usecols=(0), skiprows= 1, unpack =True)
EzI    = np.loadtxt("2_42_48um/Ref/EzRef.txt", usecols=(1), skiprows= 1, unpack =True)

ExTR   = np.loadtxt("2_42_48um/Ref/ExTra.txt", usecols=(0), skiprows= 1, unpack =True)
ExTI   = np.loadtxt("2_42_48um/Ref/ExTra.txt", usecols=(1), skiprows= 1, unpack =True)

EyTR   = np.loadtxt("2_42_48um/Ref/EyTra.txt", usecols=(0), skiprows= 1, unpack =True)
EyTI   = np.loadtxt("2_42_48um/Ref/EyTra.txt", usecols=(1), skiprows= 1, unpack =True)

EzTR   = np.loadtxt("2_42_48um/Ref/EzTra.txt", usecols=(0), skiprows= 1, unpack =True)
EzTI   = np.loadtxt("2_42_48um/Ref/EzTra.txt", usecols=(1), skiprows= 1, unpack =True)

ExIncR = np.loadtxt("2_42_48um/Ref/ExInc.txt", usecols=(0), skiprows= 1, unpack =True)
ExIncI = np.loadtxt("2_42_48um/Ref/ExInc.txt", usecols=(1), skiprows= 1, unpack =True)

EyIncR = np.loadtxt("2_42_48um/Ref/EyInc.txt", usecols=(0), skiprows= 1, unpack =True)
EyIncI = np.loadtxt("2_42_48um/Ref/EyInc.txt", usecols=(1), skiprows= 1, unpack =True)

EzIncR = np.loadtxt("2_42_48um/Ref/EzInc.txt", usecols=(0), skiprows= 1, unpack =True)
EzIncI = np.loadtxt("2_42_48um/Ref/EzInc.txt", usecols=(1), skiprows= 1 , unpack =True)

ExRef = ExR + 1j*ExI
EyRef = EyR + 1j*EyI
EzRef = EzR + 1j*EzI

ExTra = ExTR + 1j*ExTI
EyTra = EyTR + 1j*EyTI
EzTra = EzTR + 1j*EzTI

ExInc = (ExIncR + 1j*ExIncI)
EyInc = (EyIncR + 1j*EyIncI)
EzInc = (EzIncR + 1j*EzIncI)

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
			REF2_42[ff] =  REF2_42[ff] + ref[ff, i, j]
			TRA2_42[ff] =  TRA2_42[ff] + tra[ff, i, j]

###########################################################################
###########################  Pitch = 2.48 um ##############################
###########################################################################
Nx     = 119
Ny     = 205

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

EyIncR = np.zeros((NFREQs, Nx, Ny), dtype = np.complex) 
EyIncI = np.zeros((NFREQs, Nx, Ny), dtype = np.complex)
EzIncR = np.zeros((NFREQs, Nx, Ny), dtype = np.complex)
EzIncI = np.zeros((NFREQs, Nx, Ny), dtype = np.complex)


ref   = np.zeros((NFREQs, Nx, Ny), dtype = np.double)
REF2_48um   = np.zeros(NFREQs, dtype = np.double)

tra   = np.zeros((NFREQs, Nx, Ny), dtype = np.double)
TRA2_48um   = np.zeros(NFREQs, dtype = np.double)
for i in range (0, Nx):
	m[i] = i - (Nx  - 1)/2	
print(m)
for i in range (0, Ny):
	n[i] = i - (Ny  - 1)/2

freq   = np.loadtxt("2_48um/freq.txt",  usecols=(0), skiprows= 1, unpack =True)

ExR    = np.loadtxt("2_48um/ExRef.txt", usecols=(0), skiprows= 1, unpack =True)
ExI    = np.loadtxt("2_48um/ExRef.txt", usecols=(1), skiprows= 1, unpack =True)

EyR    = np.loadtxt("2_48um/EyRef.txt", usecols=(0), skiprows= 1, unpack =True)
EyI    = np.loadtxt("2_48um/EyRef.txt", usecols=(1), skiprows= 1, unpack =True)

EzR    = np.loadtxt("2_48um/EzRef.txt", usecols=(0), skiprows= 1, unpack =True)
EzI    = np.loadtxt("2_48um/EzRef.txt", usecols=(1), skiprows= 1, unpack =True)

ExTR   = np.loadtxt("2_48um/ExTra.txt", usecols=(0), skiprows= 1, unpack =True)
ExTI   = np.loadtxt("2_48um/ExTra.txt", usecols=(1), skiprows= 1, unpack =True)

EyTR   = np.loadtxt("2_48um/EyTra.txt", usecols=(0), skiprows= 1, unpack =True)
EyTI   = np.loadtxt("2_48um/EyTra.txt", usecols=(1), skiprows= 1, unpack =True)

EzTR   = np.loadtxt("2_48um/EzTra.txt", usecols=(0), skiprows= 1, unpack =True)
EzTI   = np.loadtxt("2_48um/EzTra.txt", usecols=(1), skiprows= 1, unpack =True)

ExIncR = np.loadtxt("2_48um/ExInc.txt", usecols=(0), skiprows= 1, unpack =True)
ExIncI = np.loadtxt("2_48um/ExInc.txt", usecols=(1), skiprows= 1, unpack =True)

EyIncR = np.loadtxt("2_48um/EyInc.txt", usecols=(0), skiprows= 1, unpack =True)
EyIncI = np.loadtxt("2_48um/EyInc.txt", usecols=(1), skiprows= 1, unpack =True)

EzIncR = np.loadtxt("2_48um/EzInc.txt", usecols=(0), skiprows= 1, unpack =True)
EzIncI = np.loadtxt("2_48um/EzInc.txt", usecols=(1), skiprows= 1 , unpack =True)

ExRef = ExR + 1j*ExI
EyRef = EyR + 1j*EyI
EzRef = EzR + 1j*EzI

ExTra = ExTR + 1j*ExTI
EyTra = EyTR + 1j*EyTI
EzTra = EzTR + 1j*EzTI

ExInc = (ExIncR + 1j*ExIncI)
EyInc = (EyIncR + 1j*EyIncI)
EzInc = (EzIncR + 1j*EzIncI)

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
			REF2_48um[ff] =  REF2_48um[ff] + ref[ff, i, j]
			TRA2_48um[ff] =  TRA2_48um[ff] + tra[ff, i, j]

###########################################################################
###########################################################################
###########################################################################

fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(direction = 'in', width=2, labelsize=20)
plt.ylabel("R", fontsize = '30')   
plt.xlabel(r"$\rm Wavelength \ (\mu m)$", fontsize = '30')
plt.xlim(0,10)
plt.ylim(0,1)

plt.plot((c0/freq)*1e6, REFOur, label = r'$\rm R_{Pitch = \ 2.24 \ \um}$', color = "black", linewidth = 3)
plt.plot((c0/freq)*1e6, TRAOur, label = r'$\rm T_{Pitch = \ 2.24 \ \um}$', color = "black", linewidth = 3)

plt.plot((c0/freq)*1e6, REFp2_281, label = r'$\rm R_{Pitch = \ 2.281 \ \um}$', color = "red", linewidth = 3)
plt.plot((c0/freq)*1e6, TRAp2_281, label = r'$\rm T_{Pitch = \ 2.281 \ \um}$', color = "red", linewidth = 3)

plt.plot((c0/freq)*1e6, REFPaperAg, label = r'$\rm R_{Pitch = \ 2.36 \ \um}$', color = "green", linewidth = 3)
plt.plot((c0/freq)*1e6, TRAPaperAg, label = r'$\rm T_{Pitch = \ 2.36 \ \um}$', color = "green", linewidth = 3)

plt.plot((c0/freq)*1e6, REF2_42, label = r'$\rm R_{Pitch = \ 2.42 \ \um}$', color = "blue", linewidth = 3)
plt.plot((c0/freq)*1e6, TRA2_42, label = r'$\rm T_{Pitch = \ 2.42 \ \um}$', color = "blue", linewidth = 3)

plt.plot((c0/freq)*1e6, REF2_48um, label = r'$\rm R_{Pitch = \ 2.48 \ \um}$', color = "purple", linewidth = 3)
plt.plot((c0/freq)*1e6, TRA2_48um, label = r'$\rm T_{Pitch = \ 2.48 \ \um}$', color = "purple", linewidth = 3)
# plt.plot((c0/freq)*1e6, REFScan, label = r'$\rm R_{XFDTD \ dz = 1.0 \ nm}$', color = "green", linewidth = 3)
# plt.plot((c0/freq)*1e6, REF2um, label = r'$\rm R_{XFDTD \ dz = 2.0 \ nm}$', color = "blue", linewidth = 3)



# plt.plot((c0/freq)*1e6, TRA, label = r'$\rm T_{XFDTD}$', color = "black", linewidth = 4)
# plt.plot((c0/freq)*1e6, (1-(REF+TRA)), label = r'$\rm A_{XFDTD}$', color = "limegreen", linewidth = 2)


plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(left = False, bottom = False)   
ax.legend(loc='center right', fontsize='22')
# ax.axhline(y =1, color = 'black')
# ax.axvline(x =0.1, color = 'black')
# plt.savefig("XFAGFilmOnSiDrD3x3Sweep.pdf")
plt.savefig("hBN1nmCell.png")
plt.clf()

fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(direction = 'in', width=2, labelsize=20)
plt.ylabel("R", fontsize = '30')   
plt.xlabel(r"$\rm Wavelength \ (\mu m)$", fontsize = '30')
plt.xlim(5,9)
plt.ylim(0,1)

# plt.plot(1/(k00*1e-6), Rm, label=r'$\rm R_{TM}$', color='black', linewidth = 6)
# plt.plot(1/(k00*1e-6), Tm, label=r'$\rm T_{TM}$', color='black', linewidth = 6)

plt.plot((c0/freq)*1e6, REFOur, label = r'$\rm R_{Pitch = \ 2.24 \ \um}$', color = "black", linewidth = 3)
plt.plot((c0/freq)*1e6, TRAOur, label = r'$\rm T_{Pitch = \ 2.24 \ \um}$', color = "black", linewidth = 3)

plt.plot((c0/freq)*1e6, REFp2_281, label = r'$\rm R_{Pitch = \ 2.281 \ \um}$', color = "red", linewidth = 3)
plt.plot((c0/freq)*1e6, TRAp2_281, label = r'$\rm T_{Pitch = \ 2.281 \ \um}$', color = "red", linewidth = 3)

plt.plot((c0/freq)*1e6, REFPaperAg, label = r'$\rm R_{Pitch = \ 2.36 \ \um}$', color = "green", linewidth = 3)
plt.plot((c0/freq)*1e6, TRAPaperAg, label = r'$\rm T_{Pitch = \ 2.36 \ \um}$', color = "green", linewidth = 3)

plt.plot((c0/freq)*1e6, REF2_42, label = r'$\rm R_{Pitch = \ 2.42 \ \um}$', color = "blue", linewidth = 3)
plt.plot((c0/freq)*1e6, TRA2_42, label = r'$\rm T_{Pitch = \ 2.42 \ \um}$', color = "blue", linewidth = 3)

plt.plot((c0/freq)*1e6, REF2_48um, label = r'$\rm R_{Pitch = \ 2.48 \ \um}$', color = "purple", linewidth = 3)
plt.plot((c0/freq)*1e6, TRA2_48um, label = r'$\rm T_{Pitch = \ 2.48 \ \um}$', color = "purple", linewidth = 3)
# plt.plot((c0/freq)*1e6, REFScan, label = r'$\rm R_{XFDTD \ dz = 1.0 \ nm}$', color = "green", linewidth = 3)
# plt.plot((c0/freq)*1e6, REF2um, label = r'$\rm R_{XFDTD \ dz = 2.0 \ nm}$', color = "blue", linewidth = 3)



# plt.plot((c0/freq)*1e6, TRA, label = r'$\rm T_{XFDTD}$', color = "black", linewidth = 4)
# plt.plot((c0/freq)*1e6, (1-(REF+TRA)), label = r'$\rm A_{XFDTD}$', color = "limegreen", linewidth = 2)


plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(left = False, bottom = False)   
ax.legend(loc='center right', fontsize='22')
# ax.axhline(y =1, color = 'black')
# ax.axvline(x =0.1, color = 'black')
# plt.savefig("XFAGFilmOnSiDrD3x3Sweep.pdf")
plt.savefig("hBN1nmCell5_9.png")




# plt.show()
