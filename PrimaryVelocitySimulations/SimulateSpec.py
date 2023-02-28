import numpy as np
import sys
#sys.path.insert(0, '/home/cbuzard/Github/CrossCorrelation/')
#import CCfunc_v5 as cc
sys.path.insert(0, '/home/cbuzard/Pipeline/02_PCA/')
#import AtmModel_cb as atm
import AtmModel as atm
reload(atm)
import random
from random import shuffle
import matplotlib.pyplot as plt
import os
import datetime
import math
from PyAstronomy import pyasl

def CalculateVelocity(Kp=None,P=None,tobs=None,tstart=None,e=None,arg_peri=None,f=None,M=None,Version='circular'):
	if Version == 'circular':
		return Kp*np.sin(2.*math.pi/P*((tobs-tstart)%P))
	elif Version == 'eccentric':
		return -1*Kp*(np.cos(f+(arg_peri*math.pi/180.))+e*np.sin(arg_peri*math.pi/180.))
	elif Version == 'fromM':
		return Kp*np.sin(2.*np.pi*M)	
		
	
def checkincreasing(wave,flux):
	if wave[1]<wave[0]:		### make sure it's still increasing
		wave, flux = wave[::-1],flux[::-1]
	return wave,flux

def calcJuliandate(year=None, month=None, day=None, hour=None, minute=None, DateString=None):
	## http://aa.usno.navy.mil/faq/docs/JD_Formula.php
	
	if DateString:
		fmt = '%Y%m%d_%H%M'
		dt = datetime.datetime.strptime(DateString, fmt)
		year, month, day, hour, minute = dt.year, dt.month, dt.day, dt.hour, dt.minute
	
	mintohour = minute/60.
	time = hour+mintohour
	
	term1 = 367.*year
	term2 = int((7*(year+int((month+9)/12.)))/4)
	term3 = int((275.*month)/9)
	term4 = day + 1721013.5 + time/24.
	
	full = term1 - term2 + term3 + term4
	
	return full


def ChooseM(number,method='even',plot=False):
	th = []
	if method=='random':
		for i in range(number+1):
			th.append(random.random()*2*math.pi)
	if method == 'even':
		th = np.linspace(0,2*math.pi,number+1)
		th = th + th[1]/4.		### so doesn't start at 0 (or have 0.5)
	
	if method == 'chosen':
		th = np.array([0.2,0.3,0.7,0.75,0.8,0])		### need throw away number at the end	
		th = th*math.pi*2.	
			
	th = np.array(th)[:-1]
	
	if plot:
		a = 0.0426
		e = 0
		
		n = 600.
		theta = np.arange(n)/n*2*math.pi 

		E_pirad, f_pirad = CalculateAnomalies(theta, e, showplot=False)
		f_rad = [f*math.pi for f in f_pirad]
		

		
		x, y = CalculateOrbit(a, e, f_rad)
		
		E_pirad_obs, f_pirad_obs = CalculateAnomalies(th, e, showplot=False)
		f_rad_obs = [f*math.pi for f in f_pirad_obs]

		x_obs, y_obs = CalculateOrbit(a, e, f_rad_obs)
		
		plt.figure(figsize=(8,8))
		plt.plot(x, y, 'k', linewidth=2)
		plt.plot([0,0], [0,0], '*r', markersize=25)
		#plt.plot([0,0], [a, a], 'xk', markersize=15)
		#plt.text(0, a+0.0025, '$\phi$ = 0')
		plt.plot([0,a*(1-e)],[0,0], 'xk', markersize=15)
		plt.text(a*(1-e)+0.0025, -0.0025, 'Pericenter,\n$\phi$ = 0')
		for ii, xpt in enumerate(x_obs) : 
		#	compcolor = cmap(float(ii) / np.size(M_phase_obs)+0.05)
			plt.plot([xpt,xpt], [y_obs[ii], y_obs[ii]], 'o', markersize = 15)	#, color = compcolor, label=true_dates[ii])
		#	plt.text(xpt+0.0025, y_obs[ii]-0.00125, '$\phi$ = ' + str(round(M_phase_obs[ii],2)))
		#	plt.text(xpt+0.0025, y_obs[ii]-0.004, '$v_{sec}$ = ' + str(round(vpl_kms_obs[ii], 2))+' km/s')
		leg2 = plt.legend(numpoints=1, loc=9, ncol=3, bbox_to_anchor=(0.5, 1.125))
		plt.show()
	
	return th/2/math.pi

def MakeFluxArraySmaller(wave,flux,min_wavelength,max_wavelength,wave2=None,flux2=None,edge=0):
	## be careful with this, gives fairly significant edge on your wavelength range... i'll rewrite after lunch with extra keyword
	
	min_wavelength = min_wavelength - edge				
	max_wavelength = max_wavelength + edge
	
	wave_comp = wave
	sel = np.where(wave_comp < np.min(min_wavelength))
	wave_lim = np.delete(wave_comp,sel)
	flux_lim = np.delete(flux,sel)
	if flux2 is not None:
		flux_lim2 = np.delete(flux2,sel)
	if wave2 is not None:
		wave_lim2 = np.delete(wave2,sel)	
	sel2 = np.where(wave_lim > np.max(max_wavelength))
	wave_lim = np.delete(wave_lim,sel2)
	flux_lim = np.delete(flux_lim,sel2)
	if flux2 is not None:
		flux_lim2 = np.delete(flux_lim2,sel2)
	if wave2 is not None:
		wave_lim2 = np.delete(wave_lim2,sel2)
	
	if flux2 is None and wave2 is None:
		return wave_lim, flux_lim
	elif flux2 is not None and wave2 is None:
		return wave_lim, flux_lim, flux_lim2
	elif flux2 is None and wave2 is not None:
		return wave_lim, flux_lim, wave_lim2
	elif flux2 is not None and wave2 is not None:	
		return wave_lim, flux_lim, wave_lim2, flux_lim2

def shift(wave, vel):
	### wave should be a list
	wave = list(wave)
	
	c = 2.99792458 * 10**5			### speed of light in km/s
	
	v_c = vel/c
	delwave = []
	for i in wave:
		delwave.append(i*v_c)
	
	#### assume neg velocity is toward
	#### delwave = new_wave - orig_wave
	new_wave = []
	for orig,change in zip(wave,delwave):
		new_wave.append(change+orig)	
	
	return new_wave					

def SaveTxtFiles(col1,col2,filename):
	header=[str(0), str(0), str(0), filename]
	f=open(filename, 'w')
	f.write(header[0]+'\t'+header[1]+'\t'+header[2]+'\t'+header[3] +'\n')
	for col_one, col_two in zip(col1, col2) : 
		f.write('%-10f %10f' % (col_one, col_two)+'\n')		
	f.close()
		
def Combine_Spectra(planetwave, planetflux, starwave, starflux, planetvelocity=0.0, starvelocity=0.0, band='L'):		
	#, contrast="FromModels", ,Method="2ndorder",savestar=False, stardir='/home/cbuzard/StellarModels/KELT2A/stellarmodel_KELT2A_L_starmodel1.dat'):

	### both in increasing microns		
	shiftedplanetwave = shift(planetwave, planetvelocity)
	shiftedstarwave = shift(starwave, starvelocity)
	

	#### 2.8 - 4 um, 3rd order fit, wavenumber space
	shiftedstarwave, starflux = MakeFluxArraySmaller(shiftedstarwave, starflux, np.min(shiftedplanetwave), np.max(shiftedplanetwave))
	sharedwave = shiftedplanetwave
	starflux = np.interp(sharedwave,shiftedstarwave, starflux)
	combinedspec = planetflux + starflux
	#### get average spectroscopic contrast
	##wave, planetflux = MakeFluxArraySmaller(sharedwave, planetflux/starflux, 2.995, 3.4565)	# L
	#wave, planetflux = MakeFluxArraySmaller(sharedwave, planetflux/starflux, 2.0869, 2.242)	# K
	#plt.plot(wave,planetflux)
	#print(np.mean(planetflux))
	#assert(1==0)
	
	#print(np.max(planetflux)/starflux[np.argmax(planetflux)])
	#assert(1==0)
	
	if band == 'L':
		sharedwave, combinedflux = MakeFluxArraySmaller(sharedwave,combinedspec,2.8,4)
		sharedwave = 1e4/sharedwave		## to wavenumbers
		a = np.polyfit(sharedwave, combinedflux, 3) 
		f = np.poly1d(a)
		cont_2 = f(sharedwave)
		combinedspec = combinedflux/cont_2
		sharedwave = 1e4/sharedwave
	#### 1.9 - 2.5 um, 2rd order fit, wavenumber space
	if band == 'K':
		sharedwave, combinedflux = MakeFluxArraySmaller(sharedwave,combinedspec,1.9,2.5)
		sharedwave = 1e4/sharedwave		## to wavenumbers
		a = np.polyfit(sharedwave, combinedflux, 2) 
		f = np.poly1d(a)
		cont_2 = f(sharedwave)
		combinedspec = combinedflux/cont_2
		sharedwave = 1e4/sharedwave
	
	return sharedwave, combinedspec

def Resolution(wave,data,IP=0):	  

	### from TelCorPCA

	#terraspecip(specin, ip) :
	'''
		INPUT
	specin: input spectrum [2,n]
	ip: TERRASPEC ip coefficient array (DBLARR(9)
	ip[0]: width
	ip[1:4]: height of left satellite gaussians
	ip[5:8]: height of right satellite gaussians
	note that all 8 must be specified, even if zero.  ip[0] cannot be zero

	OUTPUT
	specout = specin*ip
	'''
	specin = np.array([wave,data])
	specin = specin.transpose()
   
	if IP[0] != 0:
		ip = IP
	else:
		ip = [0.028554, 0.000000, 0.000000, 0.000000, 0.000000, 0.200000, 0.200000, 0.010000, 0.010000]		## from 29oct2013, L band
   
		if self.Resolution != 25000:
			ip[0] = ip[0]*25000/self.Resolution
   

	# Check instrument profile
	nip=np.size(ip)
	if nip != 9 : print('IP has incorrect form')

	# Interpolate spectrum on an evenly spaced grid. 
	# Add padding necessary for Fourier convolution.
	specin[:,0] = 1.0e4 / specin[:,0]	## converts to wavenumbers
	spec_wv = specin[::-1, 0]
	spec_flux = specin[::-1, 1]	 
   
	#SaveTxtFiles(spec_wv, spec_flux, "/home/cbuzard/Code/BinProblem/pyini.dat")
   
	#assert(1==0)
	mn=np.size(spec_wv)
	mmin=np.amin(spec_wv)
	mmax=np.amax(spec_wv)
	mdelta=(mmax-mmin)/(mn-1)
	mx=np.arange(mn)*mdelta+mmin
	n=mn*2
	nexp=0
	while 2**nexp < n : nexp+=1
	npad=2**nexp
	m=np.zeros(npad)

	temp=np.interp(mx,spec_wv, spec_flux)
	m[0:np.size(temp)]=temp
   
	#print len(mx)
   
	#print npad
	#print mdelta

	# Generate the ordinate vector for the instrument profile
	#temp1=np.arange(math.ceil(npad/2.))*mdelta
	#ipx=np.r_[temp1,-temp1[::-1]]

	### CB changed to match IDL rountine exactly!!! 08/08/18
	temp1=np.arange(math.ceil(npad/2.))*mdelta
	temp2=(np.arange(math.ceil(npad/2.))+1)*mdelta*-1
	ipx=np.r_[temp1,temp2[::-1]]

	'''
	The height of the satellies (ip[1:7] are parameterized as percentages 
	is the height of the central gaussian. The height of the central gaussian 
	is determined such that the areab of the total IP is unity. ip[0] is sigma 
	for all gaussians.)
	'''

	#;;Avoid underflow errors - they are computationally expensive
	#tolerance = ALOG((MACHAR(/Double)).eps)
	tolerance=-36.043653389117154		# stolen from IDL

	# Calculate the height of the central gaussian
	h=1.0/(math.sqrt(2.0*math.pi)*ip[0]*(1.0+np.sum(ip[1:])))

	# Calculate the central gaussian
	z=-0.5*(ipx/ip[0])**2
	mask = z > tolerance
	central = np.zeros(np.size(mask))
	# changed to go faster
	#for ii in range(np.size(z)) : central[ii]=mask[ii]*h*math.exp(z[ii]*mask[ii])	  
	central[:] = mask[:]*h*np.exp(z[:]*mask[:])

	# Find satellite centers
	sc=ip[0]*math.sqrt(2.0*math.log(2.0))*np.array((-1.0,-2.0,-3.0,-4,1.0,2.0,3.0,4.0))

	# World's laziest matrix math to calculate satellite gaussians
	ipxspread=np.ones(8)
	first=np.outer(ipx,ipxspread)
	scspread=np.ones(npad)
	second=np.outer(scspread.T,sc.T)
	z=-0.5*((first-second)/ip[0])**2
	mask = z > tolerance
	npadspread=np.zeros(npad)+h
	ipspread=ip[1:]
	fake=np.outer(np.transpose(npadspread), np.transpose(ipspread))
	sat_temp=np.zeros(fake.shape)
	# changed to go faster
	#for kk in range(z.shape[0]) :
	#	 for jj in range(z.shape[1]) :
	#		 sat_temp[kk,jj]=fake[kk,jj]*mask[kk,jj]*math.exp(z[kk,jj]*mask[kk,jj])
	sat_temp[:,:] = fake[:,:]*mask[:,:]*np.exp(z[:,:]*mask[:,:])
	sat=np.sum(sat_temp,axis=1)

	# Complete instrumental profile
	ipy=central+sat
	#print ipy
		   
	#plt.plot(ipx,ipy)		## plots Gaussian kernel
	#plt.show()
   

	# Convolution of instrumental profile and model spectrum
	interim = np.fft.ifft(np.fft.fft(m)*np.fft.fft(ipy/ipy.sum(axis=0)))
	my = interim[:mn].real

	# Interpolate back onto original axis and adjust wavelength units
	broad_flux = np.interp(spec_wv,mx,my)[::-1]
	broad_wv = 1.0e4 / spec_wv[::-1]		

	return broad_wv, broad_flux

def writeips(ip,filename):
	#filename_ip = self.SavePath+self.Reduction+'IP_'+str(np.size(self.SciRanges))+'_'+self.BaseName+'_'+str(self.Order)+'.dat'
	for width in ip :
		with open(filename, 'ab') as fip : 
			fip.write('%-10f' % (width)+'\t')
	with open(filename, 'ab') as fip : 
		fip.write('\n')
	fip.close()	 


target = 'SimulateVpriFromData'


Dates = 'ChosenDates' 
#Dates = 'DataDates'

if target == 'ObservingPlans':
	FileVersions = [4]	
	Kps = [60]	

	band = 'L'

	vsys = -16.965
	period = 3.0965885			
	ra = 296.7417
	dec = 34.4197
	stellarradius = 1.143		# 1.17	#Rsun
	planetradius = 1.0 #1.3 #1.0 #1.3	#1.0	#Rjup
	Tstarts = [2454343.6765]				# [2454343.66993]	
	
	planetspec = '/home/cbuzard/PlanetModels/simulatedplanet/SCARLET/temperatures/1400_Metallicity1.0_CtoO0.54_Spectrum_FullRes_thermal.dat'	## units = W/m^2/um (1.9 - 4 microns)
	stellarspec = '/home/cbuzard/StellarModels/CB_fullstellarmodels/HD187123_full_wavenumbers.dat'			## erg/s/cm^2/cm
	## factors bring both spectra into units of W/m^3
	planetfactor = 1e6 	
	starfactor = 0.1
	
	if Dates == 'DataDates':
		s_n = [1724,1713,1283,2050,2409,2298,3417]
		#s_n = [100]*7	# 7 = # of nights
		nnights = 7
		norders = [4,4,4,4,4,6,6]

		ObsDates = ['20110521_1431','20110810_0754','20131027_0613','20131029_0543','20170907_0634','20190403_1522','20190408_1508']
	
		path = '/home/cbuzard/Pipeline/03_CrossCorr/Targets/HD187123/PCAruns/'
		datafiles = np.genfromtxt('/home/cbuzard/Pipeline/03_CrossCorr/Output/HD187123_allnightsdata/HD187123.dat',max_rows=np.sum(norders),dtype='|S')		## HD187 has 5 nights

		IPs = "PCAdir"
		ip_names = np.genfromtxt('/home/cbuzard/Pipeline/03_CrossCorr/Output/HD187123_allnightsdata/HD187123_ip_names.dat',max_rows=np.sum(norders),dtype='|S')

		nnirspec1nights = 5
		nnirspec2nights = 2


if target == 'HD88133':
	FileVersions = [2]	
	Kps = [50]	

	band = 'L'

	vsys = -3.53
	period = 3.4148674	
	ecc = 0.051076309
	arg_peri = 7.2219841
	fs = np.array([1.31484744, 4.801069  , 3.83985823, 1.14875304, 3.39632546,  5.58911743])
		
	ra = 152.53198507295417
	dec = 18.186870046967222
	stellarradius = 1.943		# 1.17	#Rsun
	planetradius = 1.0 #1.3 #1.0 #1.3	#1.0	#Rjup
	
	planetspec = '/home/cbuzard/PlanetModels/HD187123/Benneke/HD187123_noninverted_wn_check.dat'	## units = W/m^2/um (1.9 - 4 microns)
	stellarspec = '/home/cbuzard/StellarModels/CB_fullstellarmodels/HD187123_full_wavenumbers.dat'			## erg/s/cm^2/cm
	## factors bring both spectra into units of W/m^3
	planetfactor = 1e6 	#0		### SET to 0 to erase pl (setting Kp=0 is not the same thing)
	starfactor = 0.1
	
	if Dates == 'DataDates':
		s_n = [1680, 2219, 2472, 1812, 1694, 2938]

		nnights = 6
		norders = [4,4,4,4,4,4]

		ObsDates = ['20120401_0805', '20120403_0809', '20130310_0651', '20130329_0525','20140514_0706','20150408_0802']
	
		path = '/home/cbuzard/Pipeline/03_CrossCorr/Targets/HD88133/'
		datafiles = np.genfromtxt('/home/cbuzard/Pipeline/03_CrossCorr/Targets/HD88133.dat',max_rows=np.sum(norders),dtype='|S')		## HD187 has 5 nights

		IPs = "direct"
		ip_names = np.genfromtxt('/home/cbuzard/Pipeline/03_CrossCorr/Targets/HD88133_ip.dat',max_rows=np.sum(norders),dtype='|S')

		nnirspec1nights = 6
		nnirspec2nights = 0

if target == 'SimulateVpriFromData':
	FileVersions = [28]	
	Kps = [150]	

	band = 'L'

	vsys = 0.0
	#period = 3.0965885			
	ra = 296.7417
	dec = 34.4197
	stellarradius = 1.0 #1.143		# 1.17	#Rsun
	planetradius = 1.0 #1.3 #1.0 #1.3	#1.0	#Rjup
	#Tstarts = [2454343.6765]				# [2454343.66993]	
	
	#planetspec = '/home/cbuzard/PlanetModels/HD187123/Benneke/HD187123_inverted_wn_check.dat'	## units = W/m^2/um (1.9 - 4 microns)
	planetspec = '/home/cbuzard/PlanetModels/HD187123/Benneke/HD187123_noninverted_wn_check.dat'	## units = W/m^2/um (1.9 - 4 microns)
	stellarspec = '/home/cbuzard/StellarModels/CB_fullstellarmodels/HD187123_full_wavenumbers.dat'			## erg/s/cm^2/cm
	## factors bring both spectra into units of W/m^3
	planetfactor = 1e6 	
	starfactor = 0.1

	if Dates == 'ChosenDates':
		NIRSPEC = 1		# 2
		nnights = 5
		norders = [4,4,4,4,4]
		#norders = [6,6,6,6,6]
		choosevbarys = False		
		if choosevbarys :
			MSelection =  'Even'	#'Quadrature'		#'Transit'	#	
			VBarySelection = 'NearZero'	#'EvenlySpaced'	#	#'LargeValues'	#	
		else:

			Ms = np.array([0.11724216, 0.05330464, 0.98222617, 0.04005487, 0.99198014])
			### these are the big abs value v pri numbers
			vbary = np.array([48.93603849679917, 38.59594855351673, 59.182824240710914, 43.77378924156944, 51.67816321008055])
			
			
			
		#####
		print("NIRSPEC :: ", NIRSPEC)
		print("nnights :: ", nnights)
	
	path = '/home/cbuzard/Pipeline/03_CrossCorr/Targets/HD187123/PCAruns/'
	datafiles = np.genfromtxt('/home/cbuzard/Pipeline/03_CrossCorr/Output/HD187123_allnightsdata/HD187123.dat',max_rows=np.sum(norders),dtype='|S')		## HD187 has 5 nights
	IPs = "PCAdir"
	ip_names = np.genfromtxt('/home/cbuzard/Pipeline/03_CrossCorr/Output/HD187123_allnightsdata/HD187123_ip_names.dat',max_rows=np.sum(norders),dtype='|S')


if target == 'HD187123':
	FileVersions = [831]	
	Kps = [56]	

	band = 'L'

	vsys = -16.965
	period = 3.0965885			
	ra = 296.7417
	dec = 34.4197
	stellarradius = 1.143		# 1.17	#Rsun
	planetradius = 1.0 #1.3 #1.0 #1.3	#1.0	#Rjup
	Tstarts = [2454343.6765]				# [2454343.66993]	
	
	planetspec = '/home/cbuzard/PlanetModels/HD187123/Benneke/HD187123_inverted_wn_check.dat'	## units = W/m^2/um (1.9 - 4 microns)
	stellarspec = '/home/cbuzard/StellarModels/CB_fullstellarmodels/HD187123_full_wavenumbers.dat'			## erg/s/cm^2/cm
	## factors bring both spectra into units of W/m^3
	planetfactor = 1e6 	
	starfactor = 0.1
	
	if Dates == 'DataDates':
		s_n = [1724,1713,1283,2050,2409,2298,3417]
		#s_n = [100]*7	# 7 = # of nights
		nnights = 7
		norders = [4,4,4,4,4,6,6]
		#ObsDates = ['20131027_0600','20110521_1430','20131029_0600','20170907_0630','20110810_0800','20190408_1500']
		#ObsDates = ['20131027_0613','20110521_1431','20131029_0543','20170907_0634','20110810_0754','20190408_1508']
		ObsDates = ['20110521_1431','20110810_0754','20131027_0613','20131029_0543','20170907_0634','20190403_1522','20190408_1508']
	
		path = '/home/cbuzard/Pipeline/03_CrossCorr/Targets/HD187123/PCAruns/'
		datafiles = np.genfromtxt('/home/cbuzard/Pipeline/03_CrossCorr/Output/HD187123_allnightsdata/HD187123.dat',max_rows=np.sum(norders),dtype='|S')		## HD187 has 5 nights
		#path = '/home/cbuzard/Pipeline/03_CrossCorr/Targets/KELT2A/'
		#datafiles = np.genfromtxt('/home/cbuzard/Pipeline/03_CrossCorr/Output/KELT2A_withoutsincinterpolation/KELT2A.dat',max_rows=5*norders,dtype='|S')
		IPs = "PCAdir"
		ip_names = np.genfromtxt('/home/cbuzard/Pipeline/03_CrossCorr/Output/HD187123_allnightsdata/HD187123_ip_names.dat',max_rows=np.sum(norders),dtype='|S')

		nnirspec1nights = 5
		nnirspec2nights = 2

if target == 'KELT2A':
	FileVersions = [1]	
	Kps = [148]	

	band = 'L'
	
	vsys = -47.4
	period = 4.1137913		
	ra = 92.6625
	dec = 30.9572
	stellarradius = 1.836			#Rsun
	planetradius = 1.290	#Rjup
	Tstarts = [2455974.60335]			
	
	planetspec = '/home/cbuzard/PlanetModels/KELT2A/Benneke/KELT2A_inverted_wn_check.dat'	## units = W/m^2/um (1.9 - 4 microns)
	stellarspec = '/home/cbuzard/StellarModels/KELT2A/bestpractices/stellarmodel_KELT2A_1.8-2.0_fastrotbro.dat'			## erg/s/cm^2/cm
	## factors bring both spectra into units of W/m^3
	planetfactor = 1e6 	
	starfactor = 0.1
	
	if Dates == 'DataDates':
		s_n = [1476,1125,1070,650,2103,1414]
		#s_n = [100]*7	# 7 = # of nights
		nnights = 6
		norders = [4,4,4,4,4,4]
		ObsDates = ['20151201_0924','20151231_1112','20160218_0726','20161215_1429','20170210_0706','20170218_0848']
		
		path = '/home/cbuzard/Pipeline/03_CrossCorr/Targets/KELT2A/'
		datafiles = np.genfromtxt('/home/cbuzard/Pipeline/03_CrossCorr/Output/KELT2A_allnightsdata/KELT2A.dat',max_rows=np.sum(norders),dtype='|S')		## HD187 has 5 nights
		IPs = "direct"
		ip_names = np.genfromtxt('/home/cbuzard/Pipeline/03_CrossCorr/Output/KELT2A_allnightsdata/KELT2A_ip.dat',max_rows=np.sum(norders),dtype='|S')

		nnirspec1nights = 6
		nnirspec2nights = 0



print("FileVersions :: ",FileVersions)
print("vsys :: ",vsys)
#print("period :: ",period)
print("ra :: ",ra)
print("dec :: ",dec)
print("stellarradius :: ",stellarradius)
print("planetradius :: ",planetradius)
print("Kps :: ",Kps)
#print("Tstarts :: ",Tstarts)
print("band :: ",band)
print("Dates :: ",Dates)
print("Stellar Model :: ",stellarspec)
print("Planet Model :: ",planetspec)
print("star factor :: ",starfactor)
print("planet factor :: ",planetfactor)

if Dates == 'ChosenDates':
	print("NIRSPEC :: ", NIRSPEC)
	print("nnights :: ", nnights)

WriteIP = False
IPFileNameToWrite = '/home/cbuzard/Pipeline/03_CrossCorr/Output/SimulateVpriFromData_information/ips_fiveepochsNIRSPEC1.dat'



if band == 'L':		 
	for FileVersion,Kp in zip(FileVersions,Kps):		 
		if Dates == 'DataDates':
			JDs = []
			for dat in ObsDates:
				JDs.append(calcJuliandate(DateString=dat))
			JDs = np.array(JDs)
			print ("Obs Dates :: ", JDs)

			## tie vbaries to obsdates
			vbary = []
			for dat in JDs:
				vh,vb = pyasl.baryCorr(dat, ra, dec, deq=2000.0)
				vbary.append(vb)
			vbary = np.array(vbary) 
			print ("vbarys :: ", vbary)

		if Dates == 'ChosenDates':		
			if NIRSPEC == 1:
				norders = [4]*nnights 
				s_n_avg = np.average([1724,1713,1283,2050,2409])
			if NIRSPEC == 2:
				norders = [6]*nnights 
				#s_n_avg = np.average([2666,2795])
				s_n_avg = np.average([2298,3417])

			
			s_n = [s_n_avg]*nnights
			#s_n = [250]*nnights
			print("S/N :: ", s_n)

			
			
			''' This defines JDs and vbary for you '''
			if choosevbarys:
				ObsDatesBounds = ['20200101_0100','20210101_0100']
				JD_bounds = []
				for dat in ObsDatesBounds:
					JD_bounds.append(calcJuliandate(DateString=dat))
				
				JDs = np.linspace(JD_bounds[0],JD_bounds[1],10000) ### smaller num=faster, can't get as cclose to wanted value, larger num==takes longer
				## tie vbaries to obsdates
				vbary = []
				for dat in JDs:
					vh,vb = pyasl.baryCorr(dat, ra, dec, deq=2000.0)
					vbary.append(vb)
				### combine with vsys
				vbary = vsys - np.array(vbary)

				### choose what vbarys you want
				jd_nearzero = []
				vbary_nearzero = []
				if VBarySelection == 'NearZero':
					for jj,vv in enumerate(np.abs(vbary)):
						if vv<2:
							jd_nearzero.append(JDs[jj])
							vbary_nearzero.append(vbary[jj])	
				if VBarySelection == 'EvenlySpaced':
					vbarymin = np.min(vbary)
					vbarymax = np.max(vbary)
					closeenoughvbarys = np.linspace(vbarymin,vbarymax,nnights)
					for close in closeenoughvbarys:
						onesetvb,onesetjd = [],[]
						diffs = np.abs(close - vbary)
						for ind,diff in enumerate(diffs):
							if diff<0.5:
								onesetjd.append(JDs[ind])
								onesetvb.append(vbary[ind])
						jd_nearzero.append(onesetjd)
						vbary_nearzero.append(onesetvb)
						### will have to think about how this works... if even orbital phasing
						### which phase = which vbary
										
						#assert(1==0)
				if VBarySelection == 'LargeValues':
					vstarmax = np.max(np.abs(vbary))
					for jj,vv in enumerate(np.abs(vbary)):
						if vv>(vstarmax-2.):
							jd_nearzero.append(JDs[jj])
							vbary_nearzero.append(vbary[jj])
				
				#### CHOOSE ORBITAL POSITIONS
				if VBarySelection == 'NearZero' or VBarySelection == 'LargeValues':
					Ms = ((np.array(jd_nearzero)-Tstart)%period)/period
					jd_nearzero = np.array(jd_nearzero)[np.argsort(Ms)]
					Ms = np.sort(Ms)
					if MSelection == 'Even':
						if nnights == 5:
							Ms_start = [0.05,0.25,0.45,0.65,0.85] ## even phasing
						if nnights == 10:
							Ms_start = [0.09090909, 0.18181818, 0.27272727, 0.36363636, 0.45454545, 0.54545455, 0.63636364, 0.72727273, 0.81818182,  0.90909091] ## even phasing
					if MSelection == 'Transit':
						Ms_start = [0.99,0.01,0.49,0.5,0.51] ## transit phasing
					if MSelection == 'Quadrature':
						if nnights == 5:
							Ms_start = [0.24,0.26,0.74,0.75,0.76] ## quadrature phasing
						if nnights == 10:
							Ms_start = np.array([0.2532560770709638, 0.7421325831436613, 0.2482394915704285, 0.7423947005716772, 0.25517460564579036, 0.7448374640025356, 0.24810843293160992, 0.7468870512161807, 0.24374714092592498, 0.7445753465745197])		## from version 681
					Ms_final = []
					JDs_final = []
					for mstart in Ms_start:
						#for alljds,allms in enumerate(Ms):
						diffs = np.abs(mstart - np.array(Ms))
						Ms_final.append(Ms[np.argmin(diffs)])
						Mdiffs = np.abs(Ms[np.argmin(diffs)]-Ms)
						JDs_final.append(jd_nearzero[np.argmin(Mdiffs)])

				### will have to think about how this works... if even orbital phasing
				### which phase = which vbary
				elif VBarySelection == 'EvenlySpaced':	
					if MSelection == 'Even':
						if nnights == 5:
							Ms_starts = [0.6508802792940624, 0.25194549023164187, 0.8464214114013481, 0.052567300824324315, 0.44932144388390244] 	## version 669
				

						if nnights == 10:
							Ms_starts = [0.27275083230734876, 0.3652663713873026, 0.9091469069662093, 0.09365375042046388, 0.72734494422045, 0.6359615815816538, 0.8178946029662315, 0.45678079266491733, 0.5458050989027068, 0.1820701149227487]		## version 712, 754
							
					if MSelection == 'Transit':	
						Ms_starts = [0.99,0.01,0.49,0.5,0.51] ## transit phasing
					if MSelection == 'Quadrature':	
						if nnights == 5:
							Ms_starts = [0.24,0.26,0.74,0.75,0.76] ## quadrature phasing			
						if nnights == 10:
							Ms_starts = np.array([0.2532560770709638, 0.7421325831436613, 0.2482394915704285, 0.7423947005716772, 0.25517460564579036, 0.7448374640025356, 0.24810843293160992, 0.7468870512161807, 0.24374714092592498, 0.7445753465745197])		## from version 681
									  
					Ms_final = []
					JDs_final = []
					for index, jd_nz in enumerate(jd_nearzero):
						Ms = ((np.array(jd_nz)-Tstart)%period)/period
						jd_nz = np.array(jd_nz)[np.argsort(Ms)]
						Ms = np.sort(Ms)
						Ms_start = Ms_starts[index]
						diffs = np.abs(Ms_start - np.array(Ms))
						Ms_final.append(Ms[np.argmin(diffs)])
						Mdiffs = np.abs(Ms[np.argmin(diffs)]-Ms)
						JDs_final.append(jd_nz[np.argmin(Mdiffs)])								  
				
				print("Mselection :: ", MSelection)		
				print("Ms :: ", Ms_final)
				JDs = np.array(JDs_final)
				print("JDs :: ", JDs)
				vbary = []
				for dat in JDs:
						vh,vb = pyasl.baryCorr(dat,ra,dec,deq=2000.0)
						vbary.append(vb)
				
				print("Vbary selection :: ", VBarySelection)			
				#print("vstar :: ", vsys - np.array(vbary))				
				vbary = np.array(vbary)
				print("vbary :: ", vbary)
	
			


		planetwave, planetflux = np.transpose(np.genfromtxt(planetspec,skip_header=1))
		starwave, starflux = np.transpose(np.genfromtxt(stellarspec,skip_header=1))
		### convert to microns
		planetwave = 1e4/planetwave
		starwave = 1e4/starwave
		planetwave, planetflux = checkincreasing(planetwave, planetflux)
		starwave, starflux = checkincreasing(starwave, starflux)


		## get units to match and correct for areas
		stellararea = math.pi*(stellarradius*1.0)**2.0
		planetarea = math.pi*(planetradius*0.102719)**2.0
		starflux = starflux * starfactor * stellararea
		planetflux = planetflux * planetfactor * planetarea 
		
		print("S/N :: ",s_n)
		print("total S/N :: ", np.sqrt(np.sum(np.array(s_n)**2.)))

		starvelocity = vsys - vbary 
		planetvelocity = CalculateVelocity(Kp=Kp, M=Ms, Version='fromM') 
		planetvelocity = planetvelocity + starvelocity

		print("Star Velocity :: ", starvelocity)
		print("Planet Velocity :: ", planetvelocity)
		print("vbary :: ", vbary)
		#print("Ms :: ", Ms)
		
		method = 'on_data'
		if method == 'on_data':
			#count = 0
			#path = '/home/cbuzard/Pipeline/03_CrossCorr/Targets/HD187123/PCAruns/'
			if Dates == "ChosenDates":
				if NIRSPEC == 1:
					datafiles = np.genfromtxt('/home/cbuzard/Pipeline/03_CrossCorr/Output/HD187123_allnightsdata/HD187123_chosen_nirspec1.dat',max_rows=5*4,dtype='|S')	
					ip_names = np.genfromtxt('/home/cbuzard/Pipeline/03_CrossCorr/Output/HD187123_allnightsdata/HD187123_ip_names_chosen_nirspec1.dat',max_rows=5*4,dtype='|S')
				if NIRSPEC == 2: 
					datafiles = np.genfromtxt('/home/cbuzard/Pipeline/03_CrossCorr/Output/HD187123_allnightsdata/HD187123_chosen_updated.dat',max_rows=2*6,dtype='|S')	
					ip_names = np.genfromtxt('/home/cbuzard/Pipeline/03_CrossCorr/Output/HD187123_allnightsdata/HD187123_ip_names_chosen.dat',max_rows=2*6,dtype='|S')

	
			count = 0
			for night, pl in enumerate(planetvelocity): 
				for order in range(norders[night]):
					if Dates == 'ChosenDates':
						### difference here is just because N2 are PCA'd into further directory.. 
						if NIRSPEC == 1:
							print (datafiles[count % (5*4)])		## 24 = 6 nights * 4 orders of KELT L data
							datafile = datafiles[count % (5*4)]
							dataspec = np.genfromtxt(path+datafile,skip_header=1)		## so it'll work for >6 epochs, loops through ips and files 
							dataspec = np.transpose(dataspec)
							dataspec[0] = 1e4/dataspec[0]		### to microns
							dataspec = checkincreasing(dataspec[0],dataspec[1])
				
							###IPs from PCA dir
							ip_name = ip_names[count % (5*4)]
							pcadir = datafile.split('/')[0]
							fullpath = path+pcadir+'/'	#+ip_name	
							everything = os.listdir(fullpath)
							end = ip_name.split('*')[1]
							print (end)
							for ii in everything:
								if ii.endswith(end):
									ipfilename = ii
						if NIRSPEC == 2:
							print (datafiles[count % (2*6)])		## 2 nights, 6 orders
							datafile = datafiles[count % (2*6)]
							dataspec = np.genfromtxt(path+datafile,skip_header=1)		## so it'll work for >6 epochs, loops through ips and files 
							dataspec = np.transpose(dataspec)
							dataspec[0] = 1e4/dataspec[0]		### to microns
							dataspec = checkincreasing(dataspec[0],dataspec[1])
					
							###IPs from PCA dir
							ip_name = ip_names[count % (2*6)]
							pcadir = datafile.split('/')[0]+'/'+datafile.split('/')[1]+'/'+datafile.split('/')[2]
							fullpath = path+pcadir+'/'	#+ip_name	
							everything = os.listdir(fullpath)
							end = ip_name.split('*')[1]
							print (end)
							for ii in everything:
								if ii.endswith(end):
									ipfilename = ii
						
						ip = np.genfromtxt(fullpath+ipfilename)
									
					else:
						print (datafiles[count % (nnights*norders[night])])		## 24 = 6 nights * 4 orders of KELT L data
						datafile = datafiles[count % (nnights*norders[night])]
						dataspec = np.genfromtxt(path+datafile,skip_header=1)		## so it'll work for >6 epochs, loops through ips and files 
						dataspec = np.transpose(dataspec)
						dataspec[0] = 1e4/dataspec[0]		### to microns
						dataspec = checkincreasing(dataspec[0],dataspec[1])
				
						###IPs from PCA dir
						if IPs == 'PCAdir':
							ip_name = ip_names[count % (nnights*norders[night])]
							if night > nnirspec1nights - 1:
								pcadir = datafile.split('/')[0]+'/'+datafile.split('/')[1]+'/'+datafile.split('/')[2]
							else:
								pcadir = datafile.split('/')[0]
							fullpath = path+pcadir+'/'	#+ip_name	
							everything = os.listdir(fullpath)
							end = ip_name.split('*')[1]
							print (end)
							for ii in everything:
								if ii.endswith(end):
									ipfilename = ii
	
							ip = np.genfromtxt(fullpath+ipfilename)
			
						if IPs == 'direct':
							ip = [float(s) for s in ip_names[count % (nnights*norders[night])]]
			
			
					### Write IP to common file
					if WriteIP:
						writeips(ip,IPFileNameToWrite)				
				

					wave, data = Combine_Spectra(planetwave, planetflux, starwave, starflux, planetvelocity=planetvelocity[night], starvelocity=starvelocity[night])
				
					wave, data = Resolution(wave,data,IP=ip)
		
					#### add shot noise
					noisearr = 1.0/s_n[night]*np.array(data)*np.random.randn(len(data))
					data = data + noisearr	
	
					### put onto data wave axis
					data = np.interp(dataspec[0],wave,data)
					wave = dataspec[0]

					#### use nan's from dataspec
					for i,j in enumerate(np.isnan(dataspec[1])):
						if j:		## i.e. value is a nan
							data[i] = np.nan
					

					
					### convert to wavenumners, increasing wave nummber and save file
					wave = 1e4/wave
					wave, data = checkincreasing(wave,data)
					dataspec = checkincreasing(1e4/dataspec[0],dataspec[1])
	
					
					SaveFile = '/home/cbuzard/Pipeline/03_CrossCorr/Targets/'+target+'_simulations/version'+str(FileVersion)+'_night'+str(night+1)+'_order'+str(order+1)+'.dat'
					SaveTxtFiles(wave, data,SaveFile)
					print (SaveFile)
	
					count += 1

if band == 'K':
	for FileVersion,Kp,Tstart in zip(FileVersions,Kps,Tstarts):
		Dates = 'DataDates' #'DataDates'	#'ChosenDates'

		if Dates == 'DataDates':
			s_n = [1618,2288,5091,2225,2910,2206,2368]
			nnights = 7
			norders = [3,3,3,3,3,3,3]
			ObsDates = ['20150406_1551','20150507_1910','20150508_1729','20150524_1838','20160415_1622','20160425_1653','20170618_2038']
			JDs = []
			for dat in ObsDates:
				JDs.append(calcJuliandate(DateString=dat))
			JDs = np.array(JDs)
			print ("Obs Dates :: ", JDs)

			## tie vbaries to obsdates
			vbary = []
			for dat in JDs:
				vh,vb = pyasl.baryCorr(dat, ra, dec, deq=2000.0)
				vbary.append(vb)
			vbary = np.array(vbary) 
			print ("vbarys :: ", vbary)

		if Dates == 'ChosenDates':
			nepochs = 5
			nnights = nepochs
			M = ChooseM(nepochs,method="even")							#np.array([0.26,0.57,0.44,0.68,0.46,0.42])
	
			choosenewdates = False
			usepreviousdates= True
	
			if choosenewdates:
				ObsDatesBounds = ['20190101_0100','20191231_1430']
				JD_bounds = []
				for dat in ObsDatesBounds:
					JD_bounds.append(calcJuliandate(DateString=dat))
				Date_ini = np.random.uniform(JD_bounds[0],JD_bounds[1],size=nepochs)
				difffromzeros = (Date_ini - Tperi) % period		## if M's are defined from tperi
				startingpts = Date_ini - difffromzeros
	
				obsdates = []
				for i,j in enumerate(M):
					obsdates.append(period*M[i]+startingpts[i])
				obsdates = np.array(obsdates)
				JDs = obsdates
				print ("Obs Dates :: ", JDs)
		
				### what is Mprime for those Ms
				M = ((JDs-Tperi)%period)/period
				print ("M :: ", M)
				Mprime = ((JDs-Tstart)%period)/period
				print ("Mprime :: ", Mprime )
		
				## tie vbaries to obsdates
				vbary = []
				for dat in JDs:
					vh,vb = pyasl.baryCorr(dat, ra, dec, deq=2000.0)
					vbary.append(vb)
				vbary = np.array(vbary) 
				print ("vbarys :: ", vbary)
	
			if usepreviousdates:
				if nepochs == 10:
					### from version10
					JDs = np.array([2458669.67351617,  2458546.11986245,  2458822.02538993,	 2458682.98882221,
							2458562.53175129,  2458810.56803357,  2458832.55377145,	 2458829.76684693,
							2458796.01409441,  2458691.03993749])
					vbary = np.array([7.71741666,	9.34935308, -14.83300748,	4.11502452,	 13.07482032,
								-16.41527482, -12.85390544, -13.42143932, -17.4322754,	  1.82532476])			
				if nepochs == 5:	
					### from ObservingStrategy_TEST17
					JDs = np.array([2458842.15,	 2458506.75,  2458675.35,  2458729.95,	2458682.55])
					vbary = np.array([-10.66054371,	 -2.19411385,	6.21231163,	 -9.08876677,	4.23758181])
			
				### what is Mprime for those Ms
				M = ((JDs-Tperi)%period)/period
				print ("M :: ", M)	
				Mprime = ((JDs-Tstart)%period)/period
				print ("Mprime :: ", Mprime)
		
				print ("Obs Dates :: ", JDs)
				print ("vbarys :: ", vbary) 
	
			totalsn = 4190.27624388
			sn = np.sqrt(totalsn**2./nepochs)
			s_n = np.ones(nepochs)*sn
			print (s_n[0])
	

		planetspec = '/home/cbuzard/PlanetModels/HD187123/Benneke/HD187123_inverted_wn_check.dat'	## units = W/m^2/um (1.9 - 4 microns)
		stellarspec = '/home/cbuzard/StellarModels/CB_fullstellarmodels/HD187123_full_wavenumbers.dat'			## erg/s/cm^2/cm
		planetwave, planetflux = np.transpose(np.genfromtxt(planetspec,skip_header=1))
		starwave, starflux = np.transpose(np.genfromtxt(stellarspec,skip_header=1))
		### convert to microns
		planetwave = 1e4/planetwave
		starwave = 1e4/starwave
		planetwave, planetflux = checkincreasing(planetwave, planetflux)
		starwave, starflux = checkincreasing(starwave, starflux)


		## get units to match and correct for areas
		stellararea = math.pi*(stellarradius*1.0)**2.0
		planetarea = math.pi*(planetradius*0.102719)**2.0
		starflux = starflux * 0.1 * stellararea
		planetflux = planetflux * 1e6 * planetarea 

		print("total S/N :: ", np.sqrt(np.sum(np.array(s_n)**2.)))

		starvelocity = vsys - vbary 
		planetvelocity = CalculateVelocity(Kp, period, JDs, Tstart, Version='circular') 
		planetvelocity = planetvelocity + starvelocity

		print("Star Velocity :: ", starvelocity)
		print("Planet Velocity :: ", planetvelocity)

		method = 'on_data'
		if method == 'on_data':
			#count = 0
			path = '/home/cbuzard/Pipeline/03_CrossCorr/Targets/HD187123/HD187123_Kband/'
			datafiles = np.genfromtxt('/home/cbuzard/Pipeline/03_CrossCorr/Output/HD187123_allnightsdata/HD187123_K.dat',max_rows=np.sum(norders),dtype='|S')		
			ip_names = np.genfromtxt('/home/cbuzard/Pipeline/03_CrossCorr/Output/HD187123_allnightsdata/HD187123_ip_names_K.dat',max_rows=np.sum(norders),dtype='|S')
	
			count = 0
			for night, pl in enumerate(planetvelocity): 
				for order in range(norders[night]):
					print (datafiles[count % (nnights*norders[night])])		## 24 = 6 nights * 4 orders of KELT L data
					datafile = datafiles[count % (nnights*norders[night])]
					dataspec = np.genfromtxt(path+datafile,skip_header=1)		## so it'll work for >6 epochs, loops through ips and files 
					dataspec = np.transpose(dataspec)
					dataspec[0] = 1e4/dataspec[0]		### to microns
					dataspec = checkincreasing(dataspec[0],dataspec[1])
			
					###IPs from PCA dir
					ip_name = ip_names[count % (nnights*norders[night])]
					fullpath = path
					everything = os.listdir(fullpath)
					end = ip_name.split('*')[1]
					print (end)
					for ii in everything:
						if ii.endswith(end):
							ipfilename = ii

					ip = np.genfromtxt(fullpath+ipfilename)
			
					### Write IP to common file
					if WriteIP:
						writeips(ip,IPFileNameToWrite)				
				

					wave, data = Combine_Spectra(planetwave, planetflux, starwave, starflux, planetvelocity=planetvelocity[night], starvelocity=starvelocity[night],band='K')
				
					wave, data = Resolution(wave,data,IP=ip)
		
					#### add shot noise
					noisearr = 1.0/s_n[night]*np.array(data)*np.random.randn(len(data))
					data = data + noisearr	
	
					### put onto data wave axis
					data = np.interp(dataspec[0],wave,data)
					wave = dataspec[0]

					#### use nan's from dataspec
					for i,j in enumerate(np.isnan(dataspec[1])):
						if j:		## i.e. value is a nan
							data[i] = np.nan

					### convert to wavenumners, increasing wave nummber and save file
					wave = 1e4/wave
					wave, data = checkincreasing(wave,data)
					dataspec = checkincreasing(1e4/dataspec[0],dataspec[1])

			
					SaveFile = '/home/cbuzard/Pipeline/03_CrossCorr/Targets/HD187123_simulations/version'+str(FileVersion)+'_night'+str(night+1)+'_order'+str(order+1)+'.dat'
					SaveTxtFiles(wave, data,SaveFile)
					print (SaveFile)
	
					count += 1


