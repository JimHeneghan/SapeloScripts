import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import math 


Rx= np.loadtxt("Ref.txt", usecols=(0), skiprows= 1, unpack =True )
# R0 = np.loadtxt("../Vac/Ref/Ref.txt", usecols=(0,), skiprows= 1, unpack =True )
# Ix = np.loadtxt("../Vac/Inc/Inc.txt", usecols=(0,2), skiprows= 1, unpack =True )#../StubbyhBN/StubbyVac/Inc/
# Tx = np.loadtxt("../Trans/Trans.txt", usecols=(0,2), skiprows= 1, unpack =True )
# I[0:len(I)] = I[0:len(I)]*0.6
# R[0:len(R0)] = R[0:len(R0)] - R0[0:len(R0)]
# R = np.sqrt(Rx**2 + Ry**2)
# I = np.sqrt(Ix**2 + Iy**2)
# T = np.sqrt(Tx**2 + Ty**2)
plt.plot(Rx)
# plt.ylim(0,1e-5)
plt.savefig("Rx.png")
plt.clf()

plt.plot(Rx[110000:140000])
plt.savefig("R0.png")
plt.clf()
# #
# #plt.show()

# plt.plot(Ix)
# plt.savefig("Ix.png")
# plt.clf()

# plt.plot(Ix[0:5000])
# plt.savefig("Ix2.png")
# plt.clf()

plt.plot(Rx[0: 200])#[500000:520000])
plt.savefig("Rx4.png")
# plt.show()
plt.clf()

plt.plot(Rx[5000000: len(Rx)])#[500000:520000])
plt.savefig("Rx2.png")
# plt.show()
plt.clf()

# plt.plot(Tx)
# plt.savefig("Tx.png")
# plt.clf()



# plt.plot(Ry)
# # plt.ylim(0,1e-5)
# plt.savefig("Ry.png")
# plt.clf()

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

stopper = len(Rx)
timeR = np.linspace(0,len(Rx), 1)
timeI = np.linspace(0,len(Ix), 1)
time = np.linspace(0,140000)

print(len(Rx))
print(len(Ry))
c0 = 3e8
ddx = 1e-9
dt = ddx/(2*c0)
print(dt)


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

# Ixfft = np.fft.fft(Ix, len(R0tail))
# Iyfft = np.fft.fft(Iy, len(R0tail))
# Ifft  = np.fft.fft(I,  len(R0tail))

# Rxfft = np.fft.fft(Rx, len(R0tail))
# Ryfft = np.fft.fft(Ry, len(Ry))
# Rfft  = np.fft.fft(R,  len(R0tail))

# Txfft = np.fft.fft(Tx, len(R0tail))
# Tyfft = np.fft.fft(Ty, len(R0tail))
# Tfft  = np.fft.fft(T,  len(R0tail))

# fs = 1/(dt*len(R0tail))
# f = fs*np.arange(0,len(R0tail))

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

# # plt.xlim(0, 1e15)
# # plt.plot(f, Rxfft)
# # plt.savefig("Rxfft.png")
# # plt.clf()

# lam = c0/f
# fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
# plt.setp(ax.spines.values(), linewidth=2)
# plt.tick_params(direction = 'in', width=2, labelsize=20)
# plt.ylabel("T", fontsize = '30')   
# plt.xlabel(r"$\rm Wavelength \ (\mu m)$", fontsize = '30')
# plt.xlim(5, 9)
# plt.ylim(0,1)
# plt.plot(lam*1e6, (abs(Txfft)/abs(Ixfft))**2, label = "T", color = "black", linewidth = 3)
# plt.plot(lam*1e6, (abs(Rxfft)/abs(Ixfft))**2, label = "R", color = "red", linewidth = 3)
# plt.plot(lam*1e6, (abs(Txfft)/abs(Ixfft))**2 + (abs(Rxfft)/abs(Ixfft))**2, label = "R+T", color = "limegreen", linewidth = 3)

# plt.setp(ax.spines.values(), linewidth=2)
# plt.tick_params(left = False, bottom = False)   
# ax.legend(loc='center right', fontsize='30')
# ax.axvline(x = 7.295, color = 'black', linewidth = 2)
# plt.savefig("hBNxRTMyCodeAbsTop.png")
# plt.clf()
# # plt.show()

# fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
# plt.setp(ax.spines.values(), linewidth=2)
# plt.tick_params(direction = 'in', width=2, labelsize=20)
# plt.ylabel("T", fontsize = '30')   
# plt.xlabel(r"$\rm Wavelength \ (\mu m)$", fontsize = '30')
# plt.xlim(5, 9)
# plt.ylim(0,1)
# plt.plot(lam*1e6, (abs(Tyfft)/abs(Iyfft))**2, label = "T", color = "black", linewidth = 3)
# plt.plot(lam*1e6, (abs(Tyfft)/abs(Iyfft))**2 + (abs(Ryfft)/abs(Iyfft))**2, label = "R+T", color = "limegreen", linewidth = 3)
# plt.plot(lam*1e6, (abs(Ryfft)/abs(Iyfft))**2, label = "R", color = "red", linewidth = 3)
# plt.setp(ax.spines.values(), linewidth=2)
# plt.tick_params(left = False, bottom = False)   
# ax.legend(loc='center right', fontsize='30')
# ax.axvline(x = 7.295, color = 'black', linewidth = 2)
# plt.savefig("hBNyRTMyCode.png")
# plt.clf()

# fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
# plt.setp(ax.spines.values(), linewidth=2)
# plt.tick_params(direction = 'in', width=2, labelsize=20)
# plt.ylabel("T", fontsize = '30')   
# plt.xlabel(r"$\rm Wavelength \ (\mu m)$", fontsize = '30')
# plt.xlim(5, 9)
# plt.ylim(0,1)

# plt.plot(lam*1e6, (abs(Tfft)/abs(Ifft))**2, label = "T", color = "black", linewidth = 3)
# plt.plot(lam*1e6, (abs(Tfft)/abs(Ifft))**2 + (abs(Rfft)/abs(Ifft))**2, label = "R+T", color = "limegreen", linewidth = 3)
# plt.plot(lam*1e6, (abs(Rfft)/abs(Ifft))**2, label = "R", color = "red", linewidth = 3)
# plt.setp(ax.spines.values(), linewidth=2)
# plt.tick_params(left = False, bottom = False)   
# ax.legend(loc='center right', fontsize='30')
# ax.axvline(x = 7.295, color = 'black', linewidth = 2)
# plt.savefig("hBNTotRTMyCode.png")
# # plt.xlim(0, 2)
# # plt.ylim(0,2)
# # plt.plot(lam*1e6, abs(Rfft)/abs(Ifft))
# # plt.savefig("lowlam.png")


# # plt.plot(R)
# # plt.plot(I)
# # plt.show()

# # plt.plot(R0tail)
# # plt.plot(I0tail)
# # plt.show()
