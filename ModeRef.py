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
# eps1 = 1 + (wp*wp)/(w*(1j*gamma-w))
dx = 5e-8
dy = 5e-8
# n1 = np.sqrt(eps1)

# #using the equations in chapter 2.2.r of Macleod
# #assuming the impedance of free space cancels out
# #assuming the incident media is vacuum with k0 = 0
# # 
# # unlabled equation on p 38 in Macleod after eqn 2.88 
# delta1 = n1*d*(w/c0)#*2*math.pi


# # eqn 2.93 in Macleod
# #since we behin at normal incidence eta0 = y0
# eta0 = imp0
# eta1 = (n1)*imp0
# eta2 = imp0
# Y =  (eta2*cos(delta1) + 1j*eta1*sin(delta1))/(cos(delta1) + 1j*(eta2/eta1)*sin(delta1))

# Rm = abs(((eta0 - Y)/(eta0 + Y))*conj((eta0 - Y)/(eta0 + Y)))

# #Calculating the T

# # Calculate (Power) Transmision from the result of problem 5.11 
# # from: http://eceweb1.rutgers.edu/~orfanidi/ewa/ch05.pdf
# # Note j --> -i Convention in formula below

# B = cos(delta1) + 1j*(eta2/eta1)*sin(delta1)
# C = eta2*cos(delta1) + 1j*eta1*sin(delta1)
# Tm = 4*eta0*real(eta2)/((eta0*B + C)*conj(eta0*B + C))



Nx = 45
Ny = 79
NFREQs = 200
nref = 1.0
ntra = 3.42

freq = np.zeros(NFREQs, dtype = np.double)
m  = np.zeros(Nx, dtype = np.int)
n  = np.zeros(Ny, dtype = np.int)

kx = np.zeros(Nx, dtype = np.double)
ky = np.zeros(Ny, dtype = np.double)

Sxr   = np.zeros((Nx, Ny), dtype = np.complex)
Syr   = np.zeros((Nx, Ny), dtype = np.complex)

kzR    = np.zeros((Nx, Ny), dtype = np.complex)
kzT    = np.zeros((Nx, Ny), dtype = np.complex)

ExRefI = np.zeros((Nx, Ny), dtype = np.double)
ExRefR = np.zeros((Nx, Ny), dtype = np.double)
ExRef  = np.zeros((Nx, Ny), dtype = np.complex)

EyRefI = np.zeros((Nx, Ny), dtype = np.double)
EyRefR = np.zeros((Nx, Ny), dtype = np.double)
EyRef  = np.zeros((Nx, Ny), dtype = np.complex)

EzRefI = np.zeros((Nx, Ny), dtype = np.double)
EzRefR = np.zeros((Nx, Ny), dtype = np.double)
EzRef  = np.zeros((Nx, Ny), dtype = np.complex)

ExTra  = np.zeros((Nx, Ny), dtype = np.complex)
EyTra  = np.zeros((Nx, Ny), dtype = np.complex)
EzTra  = np.zeros((Nx, Ny), dtype = np.complex)

ExInc  = np.zeros((Nx, Ny), dtype = np.complex)
EyInc  = np.zeros((Nx, Ny), dtype = np.complex)
EzInc  = np.zeros((Nx, Ny), dtype = np.complex)

Esr    = np.zeros((Nx, Ny), dtype = np.complex)


ref    = np.zeros((NFREQs, Nx, Ny), dtype = np.double)
REF    = np.zeros(NFREQs, dtype = np.double)

tra    = np.zeros((NFREQs, Nx, Ny), dtype = np.double)
TRA    = np.zeros(NFREQs, dtype = np.double)

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

ExIncR = np.loadtxt("../Vac/Ref/ExInc.txt", usecols=(0), skiprows= 1, unpack =True)
ExIncI = np.loadtxt("../Vac/Ref/ExInc.txt", usecols=(1), skiprows= 1, unpack =True)

EyIncR = np.loadtxt("../Vac/Ref/EyInc.txt", usecols=(0), skiprows= 1, unpack =True)
EyIncI = np.loadtxt("../Vac/Ref/EyInc.txt", usecols=(1), skiprows= 1, unpack =True)

EzIncR = np.loadtxt("../Vac/Ref/EzInc.txt", usecols=(0), skiprows= 1, unpack =True)
EzIncI = np.loadtxt("../Vac/Ref/EzInc.txt", usecols=(1), skiprows= 1, unpack =True)

for ff in range (0, NFREQs):
	# print(ff)
	# print("\n \n")
	for i in range (0, Nx):
		
			# print((ff*Ny + j)*Nx + i)
			ExRef[i] = (ExR[i*Ny: i*Ny + Ny] + 1j*ExI[i*Ny: i*Ny + Ny])#*(ExR[(ff*Ny + j)*Nx + i] - 1j*ExI[(ff*Ny + j)*Nx + i])
			EyRef[i] = (EyR[i*Ny: i*Ny + Ny] + 1j*EyI[i*Ny: i*Ny + Ny])#*(EyR[(ff*Ny + j)*Nx + i] - 1j*EyI[(ff*Ny + j)*Nx + i])
			EzRef[i] = (EzR[i*Ny: i*Ny + Ny] + 1j*EzI[i*Ny: i*Ny + Ny])#*(EzR[(ff*Ny + j)*Nx + i] - 1j*EzI[(ff*Ny + j)*Nx + i])

			ExTra[i] = (ExTR[i*Ny: i*Ny + Ny] + 1j*ExTI[i*Ny: i*Ny + Ny])#*(ExTR[(ff*Ny + j)*Nx + i] - 1j*ExTI[(ff*Ny + j)*Nx + i])
			EyTra[i] = (EyTR[i*Ny: i*Ny + Ny] + 1j*EyTI[i*Ny: i*Ny + Ny])#*(EyTR[(ff*Ny + j)*Nx + i] - 1j*EyTI[(ff*Ny + j)*Nx + i])
			EzTra[i] = (EzTR[i*Ny: i*Ny + Ny] + 1j*EzTI[i*Ny: i*Ny + Ny])#*(EzTR[(ff*Ny + j)*Nx + i] - 1j*EzTI[(ff*Ny + j)*Nx + i])

			ExInc[i] = (ExIncR[i*Ny: i*Ny + Ny] + 1j*ExIncI[i*Ny: i*Ny + Ny])#*(ExIncR[(ff*Ny + j)*Nx + i] - 1j*ExIncI[(ff*Ny + j)*Nx + i])
			EyInc[i] = (EyIncR[i*Ny: i*Ny + Ny] + 1j*EyIncI[i*Ny: i*Ny + Ny])#*(EyIncR[(ff*Ny + j)*Nx + i] - 1j*EyIncI[(ff*Ny + j)*Nx + i])
			EzInc[i] = (EzIncR[i*Ny: i*Ny + Ny] + 1j*EzIncI[i*Ny: i*Ny + Ny])#*(EzIncR[(ff*Ny + j)*Nx + i] - 1j*EzIncI[(ff*Ny + j)*Nx + i])

	lam = c0/freq[ff]
	# print(lam)
	k0 = 2*math.pi/lam
	kzinc = k0*nref
	kx = -2*math.pi*m/(Nx*dx)
	ky = -2*math.pi*n/(Ny*dy)
	for j in range (0, Ny):
		for i in range (0, Nx):
			kzR[i, j] = cmath.sqrt((k0*nref)**2 - kx[i]**2 - ky[j]**2)
			kzT[i, j] = cmath.sqrt((k0*ntra)**2 - kx[i]**2 - ky[j]**2)

			Esr[i, j] = cmath.sqrt(ExInc[i, j]**2 + EyInc[i, j]**2 + EzInc[i, j]**2)


	Sxr = ExRef/Esr
	Syr = EyRef/Esr
	Szr = EzRef/Esr

	Sxt = ExTra/Esr
	Syt = EyTra/Esr
	Szt = EzTra/Esr
	
	Sxrfft = fftshift(fft2(Sxr))/(Nx*Ny)
	Syrfft = fftshift(fft2(Syr))/(Nx*Ny)
	Szrfft = fftshift(fft2(Szr))/(Nx*Ny)

	Sxtfft = fftshift(fft2(Sxt))/(Nx*Ny)
	Sytfft = fftshift(fft2(Syt))/(Nx*Ny)
	Sztfft = fftshift(fft2(Szt))/(Nx*Ny)

	Sref = abs(Sxrfft)**2 + abs(Syrfft)**2 + abs(Szrfft)**2
	Stra = abs(Sxtfft)**2 + abs(Sytfft)**2 + abs(Sztfft)**2

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
# plt.xlim(2,5)
plt.ylim(1.25e8,1.4e8)

# plt.plot((c0/freq)*1e6, REF, label = r'$\rm R_{FDTD \ Ag Pattern \ 50x50x5 \ nm^{3}}$', color = "red", linewidth = 6)
plt.plot((c0/freq)*1e6, TRA, label = r'$\rm T_{FDTD \ Ag Pattern \ 50x50x5 \ nm^{3}}$', color = "black", linewidth = 6)

# plt.plot((c0*2*math.pi/(w))*1e6, Rm, label=r'$\rm R_{TM}$', color='yellow', linewidth = 2))

plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(left = False, bottom = False)   
ax.legend(loc='center right', fontsize='22')
ax.axhline(y =1, color = 'black')
ax.axvline(x =1, color = 'black')
plt.savefig("PlaneWaveAgSiTJuly31_3.pdf")
#plt.savefig("MyCodeRefJuly8_28.png")

# plt.show()
