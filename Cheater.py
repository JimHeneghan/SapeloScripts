import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import math 


Rx= np.loadtxt("Ref.txt", usecols=(0), skiprows= 1, unpack =True )
# R0 = np.loadtxt("../Vac/Ref/Ref.txt", usecols=(0,), skiprows= 1, unpack =True )
# Ix = np.loadtxt("../Inc/Inc.txt", usecols=(0), skiprows= 1, unpack =True )#../StubbyhBN/StubbyVac/Inc/
# Tx = np.loadtxt("../Trans/Trans.txt", usecols=(0), skiprows= 1, unpack =True )

timeR = np.linspace(0,len(Rx), len(Rx))
c0 = 3e8
ddx = 1e-9
dt = ddx/(2*c0)
Nlam = 250.0

arg = ((timeR*math.pi)/(3*Nlam) - 10.0)
arg = arg*arg
Ix = np.exp(-0.5*arg)*np.cos((timeR*math.pi/Nlam) - 30.0)

t = timeR*dt
lam1 = 5.0e-7
omega = 2*math.pi*c0/lam1
twidth = 3.0/omega
t0 = 10.0*twidth 
Ix2 = np.exp(-t*t/2/twidth/twidth)*np.cos(omega*t)


plt.plot(Ix)
plt.xlim(0, 10000)
plt.savefig("IxCHEAT.png")
plt.clf()

plt.plot(Ix2)
plt.xlim(0, 10000)
plt.savefig("Ix2CHEAT.png")
plt.clf()
# fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
# plt.setp(ax.spines.values(), linewidth=2)
# plt.tick_params(direction = 'in', width=2, labelsize=20)
# plt.ylabel("Field Strength", fontsize = '30')   
# plt.xlabel(r"$\rm Time Steps$", fontsize = '30')
# plt.xlim(0, 50000)
# plt.plot(Rx, label = "R", color = "red")#[500000:520000])
# plt.plot(Ix, label = "I", color = "blue")
# plt.plot(Tx, label = "T", color = "black")

# plt.setp(ax.spines.values(), linewidth=2)
# plt.tick_params(left = False, bottom = False)   
# ax.legend(loc='center right', fontsize='30')
# plt.savefig("All.png")
# # plt.show()
# plt.clf()

# plt.plot(Rx[95000: len(Rx)])#[500000:520000])
# plt.savefig("Rx2.png")
# # plt.show()
# plt.clf()

# plt.plot(Tx)
# plt.savefig("Tx.png")
# plt.clf()


# stopper = len(Rx)
# timeR = np.linspace(0,len(Rx), 1)
# timeI = np.linspace(0,len(Ix), 1)

# print(len(Rx))
# print(len(Ry))
c0 = 3e8
ddx = 1e-9
dt = ddx/(2*c0)
# print(dt)


# R0tail = np.zeros(stopper)
# R0tail[0:stopper] = Rx[0:stopper]
# plt.plot(R0tail)
# plt.savefig("Rtail.png")
# plt.clf()

# I0tail =  np.zeros(stopper)
# I0tail[0:stopper] = Ix[0:stopper]
# plt.plot(I0tail)
# plt.savefig("Itail.png")
# plt.clf()

Ixfft = np.fft.fft(Ix, len(Ix))
Ix2fft = np.fft.fft(Ix2, len(Ix2))
# # Iyfft = np.fft.fft(Iy, len(R0tail))
# # Ifft  = np.fft.fft(I,  len(R0tail))

Rxfft = np.fft.fft(Rx, len(Rx))
# # Ryfft = np.fft.fft(Ry, len(Ry))
# # Rfft  = np.fft.fft(R,  len(R0tail))

# Txfft = np.fft.fft(Tx, len(R0tail))
# # Tyfft = np.fft.fft(Ty, len(R0tail))
# # Tfft  = np.fft.fft(T,  len(R0tail))

fs = 1/(dt*len(Rx))
f = fs*np.arange(0,len(Rx))

# # Reflectivity = Rfft/Ifft
# # plt.ylim(0, 2)
# # plt.xlim(0,300e12)
# # plt.plot(f, Reflectivity)
# # plt.show()

# # plt.plot(f, abs(Rfft)/abs(Ifft))
# # plt.xlim(0, 1e15)
# # plt.plot(f, Ryfft)
# # plt.savefig("Ryfft.png")
# # plt.clf()

lam = c0/f

plt.xlim(0, 10)
plt.plot(lam*1e6, Ixfft)
plt.savefig("IxfftCHEAT.png")
plt.clf()

plt.xlim(0, 10)
plt.plot(lam*1e6, Ix2fft)
plt.savefig("Ix2fftCHEAT.png")
plt.clf()

plt.xlim(0, 10)
plt.plot(lam*1e6, Rxfft)
plt.savefig("Rxfft.png")
plt.clf()

fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(direction = 'in', width=2, labelsize=20)
plt.ylabel("R", fontsize = '30')   
plt.xlabel(r"$\rm Wavelength \ (\mu m)$", fontsize = '30')
plt.xlim(0, 10)
plt.ylim(0,1) #3e-6)
# plt.plot(lam*1e6, (abs(Txfft)/abs(Ixfft))**2, label = "T", color = "black", linewidth = 3)
plt.plot(lam*1e6, (abs(Rxfft)/abs(Ixfft))**2, label = "R", color = "red", linewidth = 3)
# plt.plot(lam*1e6, (abs(Txfft)/abs(Ixfft))**2 + (abs(Rxfft)/abs(Ixfft))**2, label = "R+T", color = "limegreen", linewidth = 3)

plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(left = False, bottom = False)   
ax.legend(loc='center right', fontsize='30')
# ax.axvline(x = 7.295, color = 'black', linewidth = 2)
plt.savefig("AgMyCodeRCHEAT.png")
# plt.clf()
# # plt.sh ow()

fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(direction = 'in', width=2, labelsize=20)
plt.ylabel("R", fontsize = '30')   
plt.xlabel(r"$\rm Wavelength \ (\mu m)$", fontsize = '30')
plt.xlim(0, 10)
plt.ylim(0,1) #3e-6)
plt.plot(lam*1e6, (abs(Rxfft)/abs(Ix2fft))**2, label = "R", color = "red", linewidth = 3)

plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(left = False, bottom = False)   
ax.legend(loc='center right', fontsize='30')
# ax.axvline(x = 7.295, color = 'black', linewidth = 2)
plt.savefig("AgMyCodeR2CHEAT.png")

# plt.plot(Iy)
# plt.savefig("Iy.png")
# plt.clf()

# plt.plot(Iy[0:5000])
# plt.savefig("Iy2.png")
# plt.clf()

# plt.plot(Ry[0: 5000])#[500000:520000])
# plt.savefig("Ry3.png")
# # plt.show()
# plt.clf()

# plt.plot(Ry[10000000: len(R)])#[500000:520000])
# plt.savefig("Ry2.png")
# # plt.show()
# plt.clf()

# plt.plot(Ty)
# plt.savefig("Ty.png")
# plt.clf()