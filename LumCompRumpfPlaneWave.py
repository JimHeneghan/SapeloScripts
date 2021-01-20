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
dx = 2e-8
dy = 2e-8



Nx = 115
Ny = 199

NFREQs = 500
nref = 1.0
ntra = 1.0

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

freq   = np.loadtxt("hBN2nm_BetterMedia_freq.txt",  usecols=(0), skiprows= 1, unpack =True)

ExR    = np.loadtxt("hBN2nm_BetterMedia_ExRef.txt", usecols=(0), skiprows= 1, unpack =True)
ExI    = np.loadtxt("hBN2nm_BetterMedia_ExRef.txt", usecols=(1), skiprows= 1, unpack =True)

EyR    = np.loadtxt("hBN2nm_BetterMedia_EyRef.txt", usecols=(0), skiprows= 1, unpack =True)
EyI    = np.loadtxt("hBN2nm_BetterMedia_EyRef.txt", usecols=(1), skiprows= 1, unpack =True)

EzR    = np.loadtxt("hBN2nm_BetterMedia_EzRef.txt", usecols=(0), skiprows= 1, unpack =True)
EzI    = np.loadtxt("hBN2nm_BetterMedia_EzRef.txt", usecols=(1), skiprows= 1, unpack =True)

ExTR   = np.loadtxt("hBN2nm_BetterMedia_ExTra.txt", usecols=(0), skiprows= 1, unpack =True)
ExTI   = np.loadtxt("hBN2nm_BetterMedia_ExTra.txt", usecols=(1), skiprows= 1, unpack =True)

EyTR   = np.loadtxt("hBN2nm_BetterMedia_EyTra.txt", usecols=(0), skiprows= 1, unpack =True)
EyTI   = np.loadtxt("hBN2nm_BetterMedia_EyTra.txt", usecols=(1), skiprows= 1, unpack =True)

EzTR   = np.loadtxt("hBN2nm_BetterMedia_EzTra.txt", usecols=(0), skiprows= 1, unpack =True)
EzTI   = np.loadtxt("hBN2nm_BetterMedia_EzTra.txt", usecols=(1), skiprows= 1, unpack =True)

ExIncR = np.loadtxt("hBN2nm_BetterMedia_ExInc.txt", usecols=(0), skiprows= 1, unpack =True)
ExIncI = np.loadtxt("hBN2nm_BetterMedia_ExInc.txt", usecols=(1), skiprows= 1, unpack =True)

EyIncR = np.loadtxt("hBN2nm_BetterMedia_EyInc.txt", usecols=(0), skiprows= 1, unpack =True)
EyIncI = np.loadtxt("hBN2nm_BetterMedia_EyInc.txt", usecols=(1), skiprows= 1, unpack =True)

EzIncR = np.loadtxt("hBN2nm_BetterMedia_EzInc.txt", usecols=(0), skiprows= 1, unpack =True)
EzIncI = np.loadtxt("hBN2nm_BetterMedia_EzInc.txt", usecols=(1), skiprows= 1 , unpack =True)

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

Lambda, R, T, RT = loadtxt("hBN_Ag_Si2.31umRT.txt", usecols=(0,1,2,3), skiprows= 0, unpack =True)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(16,16),constrained_layout=True)

ax1.set_ylabel("R", fontsize = '20')   
ax1.set_xlabel(r"$\rm Wavelength \ (\mu m)$", fontsize = '20')
ax1.set_xlim(5,9)
ax1.set_ylim(0,1)

ax1.plot((c0/freq)*1e6, REF, label = r'$\rm R_{Jim \ code}$', color = "red", linewidth = 6)


ax1.plot(Lambda,R, label = r'$\rm R_{Lumerical}$', color = "crimson", linewidth = 3)

plt.setp(ax1.spines.values(), linewidth=2)
ax1.tick_params(left = False, bottom = False)   
ax1.legend(loc='upper right', fontsize='12')
ax1.axvline(x = 7.295, color = 'black', linewidth = 2)
# plt.savefig("2nmJim_vs_1nmLum_Ref.pdf")
# plt.savefig("2nmJim_vs_1nmLum_Ref.png")



#################################################################
#							Transmission						#
#################################################################

ax2.set_ylabel("R", fontsize = '20')   
ax2.set_xlabel(r"$\rm Wavelength \ (\mu m)$", fontsize = '20')
ax2.set_xlim(5,9)
ax2.set_ylim(0,1)

ax2.plot((c0/freq)*1e6, TRA, label = r'$\rm T_{Jim \ code}$', color = "black", linewidth = 4)

ax2.plot(Lambda,T, label = r'$\rm T_{Lumerical}$', color = "grey", linewidth = 3)


plt.setp(ax2.spines.values(), linewidth=2)
ax2.tick_params(left = False, bottom = False)   
ax2.legend(loc='upper right', fontsize='12')
ax2.axvline(x = 7.295, color = 'black', linewidth = 2)
# plt.savefig("2nmJim_vs_1nmLum_Tra.pdf")
# plt.savefig("2nmJim_vs_1nmLum_Tra.png")

#################################################################
#							Ref+Transmission					#
#################################################################

ax3.set_ylabel("R", fontsize = '20')   
ax3.set_xlabel(r"$\rm Wavelength \ (\mu m)$", fontsize = '20')
ax3.set_xlim(5,9)
ax3.set_ylim(0,1)

ax3.plot((c0/freq)*1e6, ((REF+TRA)), label = r'$\rm R+T_{Jim \ code}$', color = "limegreen", linewidth = 2)

ax3.plot(Lambda,RT, label = r'$\rm R+T_{Lumerical}$', color = "green", linewidth = 3)


plt.setp(ax3.spines.values(), linewidth=2)
ax3.tick_params(left = False, bottom = False)   
ax3.legend(loc='upper right', fontsize='12')
ax3.axvline(x = 7.295, color = 'black', linewidth = 2)
# plt.savefig("2nmJim_vs_1nmLum_RPlusT.pdf")
# plt.savefig("2nmJim_vs_1nmLum_RPlusT.png")

#################################################################
#							Absorption  						#
#################################################################

ax4.set_ylabel("R", fontsize = '20')   
ax4.set_xlabel(r"$\rm Wavelength \ (\mu m)$", fontsize = '20')
ax4.set_xlim(5,9)
ax4.set_ylim(0,1)

ax4.plot((c0/freq)*1e6, (1-(REF+TRA)), label = r'$\rm A_{Jim code}$', color = "blue", linewidth = 2)

ax4.plot(Lambda,(1-RT), label = r'$\rm A_{Lumerical}$', color = "navy", linewidth = 3)

plt.setp(ax4.spines.values(), linewidth=2)
ax4.tick_params(left = False, bottom = False)   
ax4.legend(loc='upper right', fontsize='12')
ax4.axvline(x = 7.295, color = 'black', linewidth = 2)

plt.savefig("2nmJim_vs_1nmLum_4Pan.pdf")
plt.savefig("2nmJim_vs_1nmLum_4Pan.png")