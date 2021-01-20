from pylab import *
from scipy.optimize import curve_fit

AgRes = [1650.7100777 , 1612.3833622 , 1584.78655619, 1549.42671212,
         1515.61010203, 1487.20999405, 1459.85337525, 1433.48664951,
         1404.49339572, 1376.653309  , 1356.48325747, 1333.69005097,
         1308.55659926, 1293.32660286, 1269.68004063, 1249.68664107,
         1227.59666771, 1214.18001991, 1193.31671043, 1175.64003313]
numPlots = 20
ShadesRed  = ['deeppink','fuchsia', 'purple', 'darkviolet', 'blue', 'dodgerblue', 'deepskyblue', 'teal', 'springgreen', 'seagreen', 'limegreen',
 'forestgreen', 'greenyellow','gold', 'orange', 'orangered', 'salmon', 'red', 'darkred', 'lightcoral']
NFREQs = 500
TroughPol  = np.zeros(numPlots, dtype = np.double)
TroughPla  = np.zeros(numPlots, dtype = np.double)
pitchThou = np.zeros((numPlots,NFREQs), dtype = np.double)

for z in range(0, numPlots):
	def fano(frq, f0, g0, q0):
	    out = (-abs(q0)*abs(g0)/2.0 + frq - f0)**2/((g0/2.0)**2 + (frq - f0)**2)
	    return out

	def gauss(f, f1, g1):
	    return exp(-(f-f1)**2/(2*g1**2))


	def fitfunc(frq, f0, g0, q0, f1, g1, q1, f2, g2, a, b, c, d):
	    out = a - b*fano(frq, f0, g0, q0) - c*fano(frq, f1, g1, q1) + d*gauss(frq, f2, g2) 
	    return out

	base = 2 + z*4
	namy = "2_%02dumPitchRTA.txt" %base
	diry = "2_%02dum/Ref/" %base
	print(diry + namy)
	lam, ref = loadtxt(diry + namy,  usecols=(0,1), skiprows= 1, unpack =True)
	pitchThou[z] = ref
	# print(ref)
	# lam = lam[280:380]
	# ref = ref[280:380]
	# print(ref)
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
	plot (freq, fit, color = 'black', linewidth = 5)

	plot(freq, ref, label = r'$\rm R_{d_{cc}\ 2.%02d \ \mu m}$' %base, color = ShadesRed[z], linewidth = 2)
	plt.savefig("DRDSweep/PoorFit/2_%02dum.png" %base)
	plt.clf()
	# show()



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
	TroughPol[z] = res1[0]
	TroughPla[z] = res1[3] 

	fit = fitfunc(freq, *res1)
	print(fitfunc(res1[3], *res1))
	    # print(wp)
	    # print(f'omega_p = {qe*wp/hbar:15.8e} rad/s')
	#     # print(f'nu_c = {qe*nuc/hbar:15.8e} rad/s')
	# for thang in range(0,len(freq)):
	# 	if(fit[thang] == res1[3]):
	# 	# if(fit[thang] >= (res1[3] - 0.01) or (fit[thang] <= (res1[3] + 0.01))):
	# 		temp = freq[thang]
	#xlim(1000, 4000)
	#ylim(0,1)
	plot (freq, fit, color = 'black', linewidth = 5)
	plot(freq, ref, label = r'$\rm R_{d_{cc}\ 2.%02d \ \mu m}$' %base, color = ShadesRed[z], linewidth = 2)
	scatter(res1[3], fitfunc(res1[3], *res1), linewidth = 3, s=55,edgecolors = 'black', c= ShadesRed[z],  zorder = 25, label = "Polariton")       

	# plot(freq, fit)
	# show()
	plt.savefig("DRDSweep/GoodFit/2_%02dum.png" %base)
	plt.clf()

lam = loadtxt(diry + namy,  usecols=(0), skiprows= 1, unpack =True)

###############################################################################
peak  = np.zeros(numPlots, dtype = np.double)
peakA = np.zeros(numPlots, dtype = np.double)

peak2  = np.zeros(numPlots, dtype = np.double)
peakA2 = np.zeros(numPlots, dtype = np.double)

PitchLeng = np.linspace(2.02, 2.78, 20)
dcc = np.zeros(numPlots, dtype = np.double)
dcc = PitchLeng
dcc1 = dcc

for i in range(0, numPlots):
    for j in range(0,NFREQs-1):
        if ((pitchThou[i,j] < pitchThou[i,j-1]) & (pitchThou[i,j+1] > pitchThou[i,j]) & ((1/(lam[j]*1e-4) < 1400)) & ((1/(lam[j]*1e-4) > 1000))):
            peak[i]  = (1/(lam[j]*1e-4))
            # print(j)
            peakA[i] = pitchThou[i,j]

print(peakA)
for i in range(0, numPlots):
    temp = 1.0
    for j in range(250,NFREQs-1):
        if ((pitchThou[i,j] < pitchThou[i,j-1]) & (pitchThou[i,j+1] > pitchThou[i,j]) & (temp > pitchThou[i,j]) & ((1/(lam[j]*1e-4) > peak[i]))):
            temp = pitchThou[i,j]
            peak2[i]  = (1/(lam[j]*1e-4))
            # print(j)
            peakA2[i] = pitchThou[i,j]
###############################################################################
dcc = AgRes
dcc1 = AgRes
dcc=np.delete(dcc,12)
peak2=np.delete(peak2,12)

dcc=np.delete(dcc,11)
peak2=np.delete(peak2,11)

dcc1=np.delete(dcc1,15)
dcc1=np.delete(dcc1,1)

###############################################################################
fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(direction = 'in', width=2, labelsize=20)
plt.ylabel(r"$\rm R \ Anti-peak \ (cm^{-1})$", fontsize = '30')   
plt.xlabel(r"$\rm Bare\ Device\ Resonance \ (cm^{-1})$", fontsize = '30')
# plt.xlim(2e-6,2.7e-6)
# plt.ylim(7.45e-6,8.3e-6)
Drdcc, freq1, freq2 = loadtxt("freqvsdcc2.txt",  usecols=(0,1,2), skiprows= 1, unpack =True)

plt.scatter(dcc , peak2,    linewidth = 3, s=55,edgecolors = 'black', c='blue', zorder = 25, label = "Polariton")       
plt.scatter(AgRes , peak,    linewidth = 3, s=55,edgecolors = 'black', c='red',  zorder = 25, label = "Plasmon")       

plt.scatter(dcc1[0:len(freq1)], freq1,    linewidth = 3, s=55,edgecolors = 'black', c='cyan',  zorder = 25, label = "Fit Polariton")       
plt.scatter(dcc1[0:len(freq1)], freq2,    linewidth = 3, s=55,edgecolors = 'black', c='fuchsia',  zorder = 25, label = "Fit Plasmon")       

# plt.scatter(AgRes, peakA2,    linewidth = 3, s=55,edgecolors = 'black', c='blue', zorder = 25, label = "Polariton")       
# plt.scatter(AgRes, peakA,    linewidth = 3, s=55,edgecolors = 'black', c='red',  zorder = 25, label = "Plasmon")       

plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(left = False, bottom = False)   
ax.legend(loc='upper left', fontsize='22')

plt.savefig("DrDTrendFitCompfrom22.png")

###############################################################################
###############################################################################
fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(direction = 'in', width=2, labelsize=20)
plt.ylabel(r"$\rm R \ Anti-peak \ (cm^{-1})$", fontsize = '30')   
plt.xlabel(r"$\rm Bare\ Device\ Resonance \ (cm^{-1})$", fontsize = '30')
# plt.xlim(2e-6,2.7e-6)
# plt.ylim(7.45e-6,8.3e-6)
Drdcc, freq1, freq2 = loadtxt("freqvsdcc2.txt",  usecols=(0,1,2), skiprows= 1, unpack =True)

plt.scatter(dcc , peak2,    linewidth = 3, s=55,edgecolors = 'black', c='blue', zorder = 25, label = "Polariton")       
plt.scatter(AgRes , peak,    linewidth = 3, s=55,edgecolors = 'black', c='red',  zorder = 25, label = "Plasmon")       

plt.scatter(AgRes, TroughPol,    linewidth = 3, s=55,edgecolors = 'black', c='cyan',  zorder = 25, label = "Fit Polariton")       
plt.scatter(AgRes, TroughPla,    linewidth = 3, s=55,edgecolors = 'black', c='fuchsia',  zorder = 25, label = "Fit Plasmon")             

# plt.scatter(AgRes, peakA2,    linewidth = 3, s=55,edgecolors = 'black', c='blue', zorder = 25, label = "Polariton")       
# plt.scatter(AgRes, peakA,    linewidth = 3, s=55,edgecolors = 'black', c='red',  zorder = 25, label = "Plasmon")       



plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(left = False, bottom = False)   
ax.legend(loc='upper left', fontsize='22')

plt.savefig("DrDTrendFitCompfromRuns.png")


