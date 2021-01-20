import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import math


R = np.loadtxt("Ref0.txt", usecols=(1,), skiprows= 0, unpack =True )
I = np.loadtxt("../Inc/Inc0.txt", usecols=(1,), skiprows= 0, unpack =True )

plt.plot(R)
plt.plot(I)
plt.show()
timeR = np.linspace(0,len(R), 1)
timeI = np.linspace(0,len(I), 1)
time = np.linspace(0,140000)
print(len(R))
c0 = 3e8
ddx = 0.01e-6
dt = ddx/(2*c0)

# tau = 50*dt
# t0 = 3*tau
# omega = 50e12*2.0*math.pi

# arg = ((time*dt - t0)/(tau)) 
# arg=arg*arg


# C1= np.cos(omega*(time*dt - t0));
# Gauss = C1*np.exp(-arg)
# #plt.xlim(0,30e13)
# plt.plot(C1*Gauss)
# plt.show()

# Gaussfft= np.fft.fft(Gauss, 10000)
R0tail =  R#np.zeros(len(R))
# R0tail[2500:len(R)] = R[2500:len(R)]


I0tail =  np.zeros(len(R))

I0tail[0:3000] = I[0:3000]

plt.plot(R0tail)
plt.plot(I0tail)
plt.show()

Ifft = np.fft.fft(I0tail, len(R0tail))
Rfft = np.fft.fft(R0tail, len(R0tail))

fs = 1/(dt*len(R))
f = fs*np.arange(0,len(R0tail))
print(dt)
# plt.xlim(0,80e14)
# plt.plot(f, Gaussfft)
# plt.show()
# # Reflectivity = Rfft/Ifft
# plt.ylim(0, 2)
# plt.xlim(0,300e12)
# plt.plot(f, Reflectivity)
# plt.show()



# plt.plot(f, abs(Rfft)/abs(Ifft))
# # plt.plot(f, Ifft)
# plt.show()

# tau = 1000*dt
# t0 = 3*tau
lam = c0/f
plt.xlim(0.2, 1.5)
plt.ylim(0,0.1)
plt.plot(lam*1e6, np.log(abs(Rfft)/abs(Ifft)))
plt.show()
# arg = ((time*dt - t0)/(tau)) 
# # arg2 = (time-500)/5
# # arg2 = arg2*arg2
# arg = arg * arg

# plt.plot(R)
# plt.plot(I)
# plt.show()





#plt.plot(time, np.exp(-arg))
#plt.show()


#plt.plot(time, C1*np.exp(-arg))
#plt.show()

