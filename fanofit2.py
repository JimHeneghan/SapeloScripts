from pylab import *
from scipy.optimize import curve_fit

def fano(frq, f0, g0, q0):
    out = (-abs(q0)*abs(g0)/2.0 + frq - f0)**2/((g0/2.0)**2 + (frq - f0)**2)
    return out

def gauss(f, f1, g1):
    return exp(-(f-f1)**2/(2*g1**2))


def fitfunc(frq, f0, g0, q0, f1, g1, q1, f2, g2, a, b, c, d):
    out = a - b*fano(frq, f0, g0, q0) - c*fano(frq, f1, g1, q1) + d*gauss(frq, f2, g2) 
    return out

shadesBlue = ['limegreen', 'springgreen', 'turquoise', 'teal', 'darkblue', 'navy', 'blue', 'dark violet']
ShadesRed  = ['deeppink','fuchsia', 'purple', 'darkviolet', 'blue', 'dodgerblue', 'deepskyblue', 'teal', 'springgreen', 'seagreen', 'limegreen', 'forestgreen', 'gold', 'orange', 'orangered', 'red', 'darkred']

lam, ref = loadtxt("2_22umPitchRTA.txt",  usecols=(0,1), skiprows= 1, unpack =True)

# lam = lam[280:380]
# ref = ref[280:380]

lam = lam[225:500]
ref = ref[225:500]
freq = 1/(lam*1e-4)
sig = ones(len(freq))

f0 =  1504
g0 =  31.35002271305636
q0 =  0.7056628753079613
f1 =  1320
g1 =  107.78912494386906
q1 =  1.2622369540105243
f2 =  200
g2 =  1000
a =  0.8380897099519079
b =  0.024172924006592423
c =  0.1218572297577878
d =  0.13490571457534964

params = [f0, g0, q0, f1, g1, q1, f2, g2, a, b, c, d]

fit = fitfunc(freq, *params)
plot(freq, ref, label = r'$\rm R_{d_{cc}\ 2.22 \ \mu m}$', color = ShadesRed[4], linewidth = 3)
plot (freq, fit)
show()



res1, res2 = curve_fit(fitfunc, freq, ref, params, maxfev=500000)
print("f0 = ",res1[0])
print("g0 = ",res1[1])
print("q0 = ",res1[2])

print("f1 = ",res1[3])
print("g1 = ",res1[4])
print("q1 = ",res1[5])

print("f2 = ",res1[6])
print("g2 = ",res1[7])

print("a = ",res1[8])
print("b = ",res1[9])
print("c = ",res1[10])
print("d = ",res1[11])

print(res1[0], res1[3])


fit = fitfunc(freq, *res1)
    # print(wp)
    # print(f'omega_p = {qe*wp/hbar:15.8e} rad/s')
    # print(f'nu_c = {qe*nuc/hbar:15.8e} rad/s')

#xlim(1000, 4000)
#ylim(0,1)

plot(freq, ref, label = r'$\rm R_{d_{cc}\ 2.22 \ \mu m}$', color = ShadesRed[4], linewidth = 3)
plot(freq, fit)
show()