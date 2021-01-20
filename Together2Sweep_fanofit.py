from pylab import *
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
ShadesRed  = ['deeppink','fuchsia', 'purple', 'darkviolet', 'blue', 'dodgerblue', 'deepskyblue', 'teal', 'springgreen', 'seagreen', 'limegreen',
 'forestgreen', 'greenyellow','gold', 'orange', 'orangered', 'salmon', 'red', 'darkred', 'lightcoral']

fig, axs = plt.subplots(5, 2, figsize = (12, 15),constrained_layout=True)
z = 10
for ax0 in axs.flat:
	base = 2 + z*4

	ax0.set_title(r"$\rm d_{cc} =  2.%02d \ \mu m $" %base, fontsize = '15')
	ax0.set_ylabel(r"$ \rm R \ $", fontsize = '10')
	ax0.set_xlabel(r"$ \rm Wavenumber \ (cm^{-1})$", fontsize = '10')
	plt.setp(ax0.spines.values(), linewidth=2)
	plt.tick_params(left = False, bottom = False)
	def fano(frq, f0, g0, q0):
	    out = (-abs(q0)*abs(g0)/2.0 + frq - f0)**2/((g0/2.0)**2 + (frq - f0)**2)
	    return out

	def gauss(f, f1, g1):
	    return exp(-(f-f1)**2/(2*g1**2))


	def fitfunc(frq, f0, g0, q0, f1, g1, q1, f2, g2, a, b, c, d):
	    out = a - b*fano(frq, f0, g0, q0) - c*fano(frq, f1, g1, q1) + d*gauss(frq, f2, g2) 
	    return out

	
	namy = "2_%02dumPitchRTA.txt" %base
	diry = "2_%02dum/Ref/" %base
	print(diry + namy)
	lam, ref = loadtxt(diry + namy,  usecols=(0,1), skiprows= 1, unpack =True)
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
	ax0.plot(freq, fit, color = 'black', linewidth = 5)

	ax0.plot(freq, ref, label = r'$\rm R_{d_{cc}\ 2.%02d \ \mu m}$' %base, color = ShadesRed[z], linewidth = 2)
	# plt.savefig("DRDSweep/PoorFit/2_%02dum.png" %base)
	# plt.clf()
	# show()



	# res1, res2 = curve_fit(fitfunc, freq, ref, params, maxfev=500000)
	# print("f0 = ",res1[0])
	# print("g0 = ",res1[1])
	# print("q0 = ",res1[2])

	# print("f1 = ",res1[3])
	# print("g1 = ",res1[4])
	# print("q1 = ",res1[5])

	# print("f2 = ",res1[6])
	# print("g2 = ",res1[7])

	# print("a = ",res1[8])
	# print("b = ",res1[9])
	# print("c = ",res1[10])
	# print("d = ",res1[11])

	# print(res1[0], res1[3])


	# fit = fitfunc(freq, *res1)
	#     # print(wp)
	#     # print(f'omega_p = {qe*wp/hbar:15.8e} rad/s')
	#     # print(f'nu_c = {qe*nuc/hbar:15.8e} rad/s')

	# #xlim(1000, 4000)
	# #ylim(0,1)
	# ax0.plot(freq, fit, color = 'black', linewidth = 5)
	# ax0.plot(freq, ref, label = r'$\rm R_{d_{cc}\ 2.%02d \ \mu m}$' %base, color = ShadesRed[z], linewidth = 2)

	z = z + 1
	# ax0.plot(freq, fit)
	# show()
	# plt.savefig("DRDSweep/GoodFit/2_%02dum.png" %base)
	# plt.clf()
plt.savefig("DRDSweep/PoorFitComp2ndHalf.png")

fig, axs = plt.subplots(5, 2, figsize = (12, 15),constrained_layout=True)
z = 10
for ax0 in axs.flat:
	base = 2 + z*4

	ax0.set_title(r"$\rm d_{cc} =  2.%02d \ \mu m $" %base, fontsize = '15')

	ax0.set_xlabel(r"$ \rm R \ $", fontsize = '10')
	ax0.set_ylabel(r"$ \rm Wavenumber \ (cm^{-1})$", fontsize = '10')
	plt.setp(ax0.spines.values(), linewidth=2)
	plt.tick_params(left = False, bottom = False)
	def fano(frq, f0, g0, q0):
	    out = (-abs(q0)*abs(g0)/2.0 + frq - f0)**2/((g0/2.0)**2 + (frq - f0)**2)
	    return out

	def gauss(f, f1, g1):
	    return exp(-(f-f1)**2/(2*g1**2))


	def fitfunc(frq, f0, g0, q0, f1, g1, q1, f2, g2, a, b, c, d):
	    out = a - b*fano(frq, f0, g0, q0) - c*fano(frq, f1, g1, q1) + d*gauss(frq, f2, g2) 
	    return out

	namy = "2_%02dumPitchRTA.txt" %base
	diry = "2_%02dum/Ref/" %base
	print(diry + namy)
	lam, ref = loadtxt(diry + namy,  usecols=(0,1), skiprows= 1, unpack =True)
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

	# fit = fitfunc(freq, *params)
	# ax0.plot(freq, fit, color = 'black', linewidth = 5)

	# ax0.plot(freq, ref, label = r'$\rm R_{d_{cc}\ 2.%02d \ \mu m}$' %base, color = ShadesRed[z], linewidth = 2)
	# plt.savefig("DRDSweep/PoorFit/2_%02dum.png" %base)
	# plt.clf()
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


	fit = fitfunc(freq, *res1)
	    # print(wp)
	    # print(f'omega_p = {qe*wp/hbar:15.8e} rad/s')
	    # print(f'nu_c = {qe*nuc/hbar:15.8e} rad/s')

	#xlim(1000, 4000)
	#ylim(0,1)
	ax0.plot(freq, fit, color = 'black', linewidth = 5)
	ax0.plot(freq, ref, label = r'$\rm R_{d_{cc}\ 2.%02d \ \mu m}$' %base, color = ShadesRed[z], linewidth = 2)

	z = z + 1
	# ax0.plot(freq, fit)
	# show()
	# plt.savefig("DRDSweep/GoodFit/2_%02dum.png" %base)
	# plt.clf()
plt.savefig("DRDSweep/GoodFitComp2ndHalf.png")