from pylab import *
hbar = 1.0545718E-34
qe = 1.602176634E-19

def epsperp(w):
    epsinf = 4.87
    S = 1.83
    w0 = 170.1e-3
    gam = 3.97e-3 #0.87e-3
    eps = epsinf + S*w0*w0/(w0*w0-1j*gam*w-w*w)
    return eps 

def epspar(w):
    epsinf = 2.95
    S = 0.61
    w0 = 92.5e-3
    gam = 0.25e-3 #3.97e-3 # 0.25e-3
    eps = epsinf + S*w0*w0/(w0*w0-1j*gam*w-w*w)
    return eps
#Incorrect silver model
def epsdrude(w):
    w0 = 7.578 # 9.6 
    nuc = 0.0644 # 22.8e-3
    eps = 1.0 + w0*w0/(w*(-1j*nuc-w))
    return eps 

epsa = 1.0
# This is only the thickness of hBN
d = 80e-7
ev = linspace(0.05,1.240,100000)
um = 1.23984193/ev
cmm1 = 8065.54429*ev
epsp = epsperp(ev)
epsl = epspar(ev)
psi = sqrt(epsl/epsp)/(1j)
epss = epsdrude(ev)



q1 = real((-psi/d)*(arctan(epsa/(epsp*psi))+arctan(epsa/(epsp*psi)) + 0*pi))
q2 = real((-psi/d)*(arctan(epsa/(epsp*psi))+arctan(epss/(epsp*psi)) + 0*pi))
q3 = real((-psi/d)*(arctan(epsa/(epsp*psi))+arctan(epss/(epsp*psi)) + 1*pi))
xlim(0,80000*2*pi)
ylim(1340,1680)

plot(q1,cmm1, c='r', label="Vacuum|hBN|Vacuum")
#plot(q2,cmm1, c='b', label="Vacuum|hBN|Vacuum")
plot(q3,cmm1, c='g', label="Vacuum|hBN|Silver")

Excite, wl = np.loadtxt("VacWaves.txt", usecols=(0,1), skiprows= 1, unpack =True )
ExciteAg, wlAg = np.loadtxt("AgWaves.txt", usecols=(0,1), skiprows= 1, unpack =True )

Excite = (1/(Excite*1e-4))
wl = (2*pi/(wl*1e-4))

ExciteAg = (1/(ExciteAg*1e-4))
wlAg = (2*pi/(wlAg*1e-4))
scatter(wlAg, ExciteAg, s=20, c='g', label='FDTD Silver Data') 
scatter(wl, Excite, s=20, c='r', label='FDTD Vacuum Data') 
# scatter([50000*2*pi], [1557.6], s=20, c='g') 
# scatter([17857*2*pi], [1557.6], s=20, c='r') 
# scatter([45455*2*pi], [1545.6], s=20, c='g') 
# scatter([23810*2*pi], [1545.6], s=20, c='r') 
# scatter([41667*2*pi], [1533.7], s=20, c='g') 
# scatter([13158*2*pi], [1533.7], s=20, c='r') 
# scatter([38462*2*pi], [1522.1], s=20, c='g') 
# scatter([13889*2*pi], [1522.1], s=20, c='r') 
# scatter([33333*2*pi], [1510.6], s=20, c='g') 
# scatter([30303*2*pi], [1499.3], s=20, c='g')  
# scatter([35714*2*pi], [1466.3], s=20, c='r') 
plt.xlabel(r'q (rad cm$^{-1}$)')
plt.ylabel(r'Frequency (cm$^{-1}$)')	 

plt.legend()
plt.savefig("DrDisp.png")