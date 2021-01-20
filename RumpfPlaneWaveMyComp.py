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
# k0 = k*1e12
c0 = 3e8
dx = 5e-8
dy = 5e-8



Nx = 45
Ny = 79

NFREQs = 500
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



ref   = np.zeros((NFREQs, Nx, Ny), dtype = np.double)
REF   = np.zeros(NFREQs, dtype = np.double)

tra   = np.zeros((NFREQs, Nx, Ny), dtype = np.double)
TRA   = np.zeros(NFREQs, dtype = np.double)
for i in range (0, Nx):
	m[i] = i - (Nx  - 1)/2	
print(m)
for i in range (0, Ny):
	n[i] = i - (Ny  - 1)/2

freq   = np.loadtxt("PropC0freq.txt",  usecols=(0), skiprows= 1, unpack =True)

ExR    = np.loadtxt("PropC0ExRef.txt", usecols=(0), skiprows= 1, unpack =True)
ExI    = np.loadtxt("PropC0ExRef.txt", usecols=(1), skiprows= 1, unpack =True)

EyR    = np.loadtxt("PropC0EyRef.txt", usecols=(0), skiprows= 1, unpack =True)
EyI    = np.loadtxt("PropC0EyRef.txt", usecols=(1), skiprows= 1, unpack =True)

EzR    = np.loadtxt("PropC0EzRef.txt", usecols=(0), skiprows= 1, unpack =True)
EzI    = np.loadtxt("PropC0EzRef.txt", usecols=(1), skiprows= 1, unpack =True)

ExTR   = np.loadtxt("PropC0ExTra.txt", usecols=(0), skiprows= 1, unpack =True)
ExTI   = np.loadtxt("PropC0ExTra.txt", usecols=(1), skiprows= 1, unpack =True)

EyTR   = np.loadtxt("PropC0EyTra.txt", usecols=(0), skiprows= 1, unpack =True)
EyTI   = np.loadtxt("PropC0EyTra.txt", usecols=(1), skiprows= 1, unpack =True)

EzTR   = np.loadtxt("PropC0EzTra.txt", usecols=(0), skiprows= 1, unpack =True)
EzTI   = np.loadtxt("PropC0EzTra.txt", usecols=(1), skiprows= 1, unpack =True)

ExIncR = np.loadtxt("PropC0ExInc.txt", usecols=(0), skiprows= 1, unpack =True)
ExIncI = np.loadtxt("PropC0ExInc.txt", usecols=(1), skiprows= 1, unpack =True)

EyIncR = np.loadtxt("PropC0EyInc.txt", usecols=(0), skiprows= 1, unpack =True)
EyIncI = np.loadtxt("PropC0EyInc.txt", usecols=(1), skiprows= 1, unpack =True)

EzIncR = np.loadtxt("PropC0EzInc.txt", usecols=(0), skiprows= 1, unpack =True)
EzIncI = np.loadtxt("PropC0EzInc.txt", usecols=(1), skiprows= 1 , unpack =True)

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
			REF[ff] =  REF[ff] + ref[ff, i, j]
			TRA[ff] =  TRA[ff] + tra[ff, i, j]




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



ref   = np.zeros((NFREQs, Nx, Ny), dtype = np.double)
REF_d   = np.zeros(NFREQs, dtype = np.double)

tra   = np.zeros((NFREQs, Nx, Ny), dtype = np.double)
TRA_d   = np.zeros(NFREQs, dtype = np.double)
for i in range (0, Nx):
	m[i] = i - (Nx  - 1)/2	
print(m)
for i in range (0, Ny):
	n[i] = i - (Ny  - 1)/2

freq   = np.loadtxt("DrDStructfreq.txt",  usecols=(0), skiprows= 1, unpack =True)

ExR    = np.loadtxt("DrDStructExRef.txt", usecols=(0), skiprows= 1, unpack =True)
ExI    = np.loadtxt("DrDStructExRef.txt", usecols=(1), skiprows= 1, unpack =True)

EyR    = np.loadtxt("DrDStructEyRef.txt", usecols=(0), skiprows= 1, unpack =True)
EyI    = np.loadtxt("DrDStructEyRef.txt", usecols=(1), skiprows= 1, unpack =True)

EzR    = np.loadtxt("DrDStructEzRef.txt", usecols=(0), skiprows= 1, unpack =True)
EzI    = np.loadtxt("DrDStructEzRef.txt", usecols=(1), skiprows= 1, unpack =True)

ExTR   = np.loadtxt("DrDStructExTra.txt", usecols=(0), skiprows= 1, unpack =True)
ExTI   = np.loadtxt("DrDStructExTra.txt", usecols=(1), skiprows= 1, unpack =True)

EyTR   = np.loadtxt("DrDStructEyTra.txt", usecols=(0), skiprows= 1, unpack =True)
EyTI   = np.loadtxt("DrDStructEyTra.txt", usecols=(1), skiprows= 1, unpack =True)

EzTR   = np.loadtxt("DrDStructEzTra.txt", usecols=(0), skiprows= 1, unpack =True)
EzTI   = np.loadtxt("DrDStructEzTra.txt", usecols=(1), skiprows= 1, unpack =True)

ExIncR = np.loadtxt("DrDStructExInc.txt", usecols=(0), skiprows= 1, unpack =True)
ExIncI = np.loadtxt("DrDStructExInc.txt", usecols=(1), skiprows= 1, unpack =True)

EyIncR = np.loadtxt("DrDStructEyInc.txt", usecols=(0), skiprows= 1, unpack =True)
EyIncI = np.loadtxt("DrDStructEyInc.txt", usecols=(1), skiprows= 1, unpack =True)

EzIncR = np.loadtxt("DrDStructEzInc.txt", usecols=(0), skiprows= 1, unpack =True)
EzIncI = np.loadtxt("DrDStructEzInc.txt", usecols=(1), skiprows= 1 , unpack =True)

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
print("Second Loop")
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
			REF_d[ff] =  REF_d[ff] + ref[ff, i, j]
			TRA_d[ff] =  TRA_d[ff] + tra[ff, i, j]

fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(direction = 'in', width=2, labelsize=20)
plt.ylabel("R", fontsize = '30')   
plt.xlabel(r"$\rm Wavelength \ (\mu m)$", fontsize = '30')
plt.xlim(5,9)
plt.ylim(0,1)


plt.plot((c0/freq)*1e6, REF, label = r'$\rm R_{Jim Struct}$', color = "red", linewidth = 6)
plt.plot((c0/freq)*1e6, TRA, label = r'$\rm T_{Jim Struct}$', color = "black", linewidth = 6)
plt.plot((c0/freq)*1e6, (1-(REF+TRA)), label = r'$\rm A_{Jim Struct}$', color = "limegreen", linewidth = 6)


plt.plot((c0/freq)*1e6, REF_d, label = r'$\rm R_{Dr D Struct}$', color = "crimson", linewidth = 3)
plt.plot((c0/freq)*1e6, TRA_d, label = r'$\rm T_{Dr D Struct}$', color = "grey", linewidth = 3)
plt.plot((c0/freq)*1e6, (1-(REF_d+TRA_d)), label = r'$\rm A_{Dr D Struct}$', color = "green", linewidth = 3)

# plt.plot((c0*2*math.pi/(w))*1e6, Rm, label=r'$\rm R_{TM}$', color='yellow', linewidth = 2))

plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(left = False, bottom = False)   
ax.legend(loc='upper right', fontsize='22')
# ax.axhline(y =1, color = 'black')
# ax.axvline(x =0.1, color = 'black')
plt.savefig("CPMLSelfAgHolesSiSubDrDStruct_Comp_5_9RTA.pdf")
plt.savefig("CPMLSelfAgHolesSiSubDrDStruct_Comp_5_9RTA.png")

# plt.show()

# plt.show()
