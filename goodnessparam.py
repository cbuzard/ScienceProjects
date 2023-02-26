import numpy as np
import matplotlib.pyplot as plt
from scipy.io.idl import readsav
from scipy.optimize import curve_fit
import scipy
from scipy import stats

type = 'noninv'
#type = 'inv'

## Save the figures of the Gaussian fits to each simulation?
SaveFigures = False


if type == "noninv":
	#specialversions = [[13,14,15], [16,17,18], [19,20,21], [22,23,24], [625,626,627]]
	#allversions = [np.linspace(125,224,100), np.linspace(225,324,100), np.linspace(25,124,100), np.linspace(325,424,100),np.linspace(625,724,100)]
	#titles = ["A: Even $v_{pri}$ spacing","B: Largest $v_{pri}$","C: $v_{pri}$ near zero","D: Smallest $v_{pri}$","E: Random $v_{pri}$"]
	
	specialversions = [[22,23,24],[13,14,15], [16,17,18], [19,20,21],  [725,726,727]]
	allversions = [np.linspace(325,424,100),np.linspace(125,224,100), np.linspace(225,324,100), np.linspace(25,124,100), np.linspace(625,724,100)]
	#titles = ["A: Smallest $v_{pri}$","B: Even $v_{pri}$ spacing","C: Largest $v_{pri}$","D: $v_{pri}$ near zero","E: Random $v_{pri}$"]
	titles = ["A: Most blue- \n shifted $v_{pri}$","B: Even $v_{pri}$ spacing","C: Most red- \n shifted $v_{pri}$","D: $v_{pri}$ near zero","E: Random $v_{pri}$"]
	
	
if type == "inv":
	specialversions = [[1,2,3], [4,5,6], [7,8,9], [10,11,12]]
	titles = ["A: Even $v_{pri}$ spacing","B: Largest $v_{pri}$","C: $v_{pri}$ near zero","D: Smallest $v_{pri}$"]


def PeakHeightOverNoise(Ks,x_all,y_all,points,colors):
	y_all = y_all - np.mean(y_all)
	y_allbelow0 = y_all[np.where(x_all < 0)]
	std_y_allbelow0 = np.std(y_allbelow0)
	y_allatpeak = np.interp(Kps,x_all,y_all)

	points.append(y_allatpeak/std_y_allbelow0)

def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2./(2.*sigma**2.))


def MakeFluxArraySmaller(wave,flux,min_wavelength,max_wavelength,edge=0):
	## be careful with this, gives fairly significant edge on your wavelength range... i'll rewrite after lunch with extra keyword
	
	min_wavelength = min_wavelength - edge				
	max_wavelength = max_wavelength + edge
	
	wave_comp = wave
	sel = np.where(wave_comp < np.min(min_wavelength))
	wave_lim = np.delete(wave_comp,sel)
	flux_lim = np.delete(flux,sel)
	
	sel2 = np.where(wave_lim > np.max(max_wavelength))
	wave_lim = np.delete(wave_lim,sel2)
	flux_lim = np.delete(flux_lim,sel2)
		
	
	return wave_lim, flux_lim
	

def PeakHeightatKp(Kps,x_all,y_all,pointswidthofpeak,pointspeakovernoise,colors,version,vprigroup,gamma,plotfits=False,returnRsq=False):	#pointsheightofpeak
	#y_all = y_all - np.mean(y_all)
	y_all = y_all - np.mean(y_all[np.where(x_all < 0)])
	try:
		startoutA = y_all[np.where(x_all==Kps)][0]
		startoutKp = Kps
		startoutsigma = 10
		popt,pcov = curve_fit(gaus,x_all,y_all,p0=[startoutA,startoutKp,startoutsigma])
					
		a_fit, mean_fit, sigma_fit = popt
		sigma_fit = np.abs(sigma_fit)
		
		if plotfits:
			plt.figure(4)
			plt.plot(x_all,y_all)
			plt.plot(x_all,gaus(x_all,*popt),label='A = {0:.3f}'.format(a_fit) + '\n $\mu$ = {0:.3f}'.format(mean_fit) + '\n $\sigma$ = {0:.3f}'.format(sigma_fit))
			plt.title(vprigroup + "; Version " + str(version) + ", $\gamma =$ {0:.3f}".format(gamma),fontsize=16)
			plt.legend(loc=2,fontsize=14)
			plt.tight_layout()
			plt.savefig("/home/cbuzard/Desktop/VpriPaper/GaussianFits/version"+str(version)+".pdf",bbox_inches='tight')
			plt.close(4)
		
		### Kp isn't in 1sigma of fit
		if (mean_fit - sigma_fit) > Kps or (mean_fit + sigma_fit) < Kps:
			#print(mean_fit-sigma_fit, mean_fit+sigma_fit)
			#pointsheightofpeak.append(0)
			pointswidthofpeak.append(0)
			pointspeakovernoise.append(0)
			colors.append('r')
			r_squared = np.nan
			
		### if its a dip rather than a peak:
		elif a_fit < 0:
			#print(mean_fit-sigma_fit, mean_fit+sigma_fit)
			#pointsheightofpeak.append(0)
			pointswidthofpeak.append(0)
			pointspeakovernoise.append(0)
			colors.append('r')
			r_squared = np.nan
	
		else:
			#pointsheightofpeak.append(a_fit)	
			pointswidthofpeak.append(sigma_fit)
			#if sigma_fit < 3:
			#	print(version)
			#	plt.figure(3)
			#	plt.plot(x_all, y_all)
			#	plt.plot(x_all,gaus(x_all,*popt),label='fit')
			#	plt.show()
			colors.append('lightskyblue')
		
			y_allbelow0 = y_all[np.where(x_all < 0)]
			std_y_allbelow0 = np.std(y_allbelow0)
			#y_allatpeak = np.interp(Kps,x_all,y_all)
			pointspeakovernoise.append(a_fit/std_y_allbelow0)
			
			###  R Squared
			#x_all, y_all = MakeFluxArraySmaller(x_all, y_all, mean_fit-2*sigma_fit, mean_fit+2*sigma_fit)
			y_fit = gaus(x_all,*popt)
			ss_res = np.sum((y_all - y_fit) ** 2.)
			ss_tot = np.sum((y_all - np.mean(y_all)) ** 2.)
			r_squared = 1.0 - (ss_res / ss_tot)
			#r_squared = ss_res 
			
			
			#if sigma_fit < 3.0782429485005167:
			#	print(a_fit/std_y_allbelow0)
			#	print(gamma)


	## too hard to fit Gaussian
	## RuntimeError: Optimal parameters not found: Number of calls to function has reached maxfev = 800.		
	except RuntimeError:
		#pointsheightofpeak.append(0)
		pointswidthofpeak.append(0)
		pointspeakovernoise.append(0)
		colors.append('r')
		r_squared = np.nan
	
	if returnRsq:
		return r_squared	
		
def DrawNIRSPECvelocityprecisionline(ax):
	IP_sig_wn = np.array([0.02963 , 0.029191, 0.035831, 0.028137, 0.029875, 0.036543,
							0.029713, 0.027756, 0.034533, 0.023169, 0.030698, 0.034967,
							0.02963 , 0.029191, 0.035831, 0.028137, 0.029875, 0.036543,
							0.029713, 0.027756, 0.034533, 0.023169, 0.030698, 0.034967,
    						0.02963 , 0.029191, 0.035831, 0.028137, 0.029875, 0.036543])
	IP_sig_wn = np.mean(IP_sig_wn)
	
	wltouse = np.mean([3.6965, 2.9330])	## microns, ranges from 2.7 - 3.4 km/s precision
	wntouse = 1e4/wltouse 
	IP_sig_wl = IP_sig_wn * (wltouse/wntouse)
	
	IP_sig_vel = (IP_sig_wl/wltouse) * (2.99792458e5)		## km/s
	
	#print(IP_sig_vel)
	
	ax.axhline(y=IP_sig_vel,ls='--',color='gray')

def ConvertTwoTailedPValuetoSigma(pvalue):
	##  a two tailed p-value is area on both tails of Gauss distribution outward
	## x-sigma
	
	innerarea = 1. - pvalue
	x = np.sqrt(2.)*scipy.special.erfinv(innerarea)
	
	return x
	

fig, bigaxes = plt.subplots(2, 5, figsize=(12,8), sharex=True, sharey='row')
#fig, bigaxes = plt.subplots(1, 5, figsize=(12,4), sharex=True, sharey='row')

Panels = "top"
if Panels == "top":
	axes = bigaxes[0]
	parameter = "PeakHeightOverNoise"
	AllRsq = []
	
	
	for num, ax in enumerate(axes):
		for kindofpoint in range(2):
		
			#points = []						###  height at 75 relative to std from -150 to 0
			pointspeakovernoise = []		### heigh of Gaussian within 1sigma of Kp over std from -150 to 0
			#pointsheightofpeak = []			###  height of Gaussian if there is one within 1sigma of Kp
			pointswidthofpeak = []			###  width of Gaussian if there is one within 1sigma of Kp
			gammas = []
			colors = []
			
			if kindofpoint == 0:
				versions = allversions[num]
				marker = 'o'
				edgecolors = 'none'
				markersize = None	
				midrsq = []
			if kindofpoint == 1:
				versions = specialversions[num]
				marker = '*'
				edgecolors = 'k'
				markersize = 200

			for version in versions:
				version = int(version)
				
				try:
					totaldir = '/home/cbuzard/Pipeline/03_CrossCorr/Output/VpriM_simulations_version'
					Kps = 75
	
					if type == 'inv':
						dir = totaldir + str(version) + '/inv/COMBO_3/'
					if type == 'noninv':
						dir = totaldir + str(version) + '/noninv/COMBO_3/'
					m = readsav(dir+'max_like_vb_VpriM_simulations.idl.sav')
				except IOError:
					totaldir = '/export/nobackup1/cbuzard/Pipeline/03_CrossCorr/Vpri_Output/VpriM_simulations_version'
					Kps = 75
	
					if type == 'inv':
						dir = totaldir + str(version) + '/inv/COMBO_3/'
					if type == 'noninv':
						dir = totaldir + str(version) + '/noninv/COMBO_3/'
					m = readsav(dir+'max_like_vb_VpriM_simulations.idl.sav')
					
				x_all, y_all = m.toplotx, m.toploty
		
				#if version == 268:
				#	plt.figure(4)
				#	plt.plot(x_all,y_all)
				#	plt.show()		
		

				#### measure gamma
				nightsdir = '/home/cbuzard/Pipeline/03_CrossCorr/Targets/VpriM_simulations/version'+str(version)+'_M_nights.dat'
				Ms = np.genfromtxt(nightsdir, skip_header=1)[0]
				gamma = np.sum(np.abs(np.sin(2*np.pi*Ms)))
				gammas.append(gamma)

				#if parameter == "PeakHeightOverNoise":
				#	#### height of point at Kp (75 km/s) over std from -150 to 0
				#	PeakHeightOverNoise(Kps,x_all,y_all,points)
				#
				#if parameter == "PeakHeightatKp" or "PeakWidthatKp":
				#	### if there is peak within 1sigma of 75, how tall is it
				Rsq = PeakHeightatKp(Kps,x_all,y_all,pointswidthofpeak,pointspeakovernoise,colors,version, titles[num], gamma, plotfits=SaveFigures, returnRsq=True)
				midrsq.append(Rsq)
				
	
			#if parameter == "PeakHeightOverNoise":
			if kindofpoint == 0:
				print(titles[num])
				numberbluepoints = len(np.where(np.array(colors)=='lightskyblue')[0])
				print(numberbluepoints)
				bluepointindices = np.where(np.array(colors)=='lightskyblue')[0]
				bluepoints = np.array(pointspeakovernoise)[bluepointindices]
				print(np.mean(bluepoints), np.std(bluepoints))
			
				
				if titles[num] == "D: $v_{pri}$ near zero":
					nearzeroAs = pointspeakovernoise
					nearzeroSigs = pointswidthofpeak
				elif titles[num] == "E: Random $v_{pri}$":	
					randomAs = pointspeakovernoise
					randomSigs = pointswidthofpeak
				#elif titles[num] == "A: Smallest $v_{pri}$":
				elif titles[num] == "A: Most blue- \n shifted $v_{pri}$":
					smallestAs = pointspeakovernoise
					smallestSigs = pointswidthofpeak
				#elif titles[num] == "C: Largest $v_{pri}$":	
				elif titles[num] == "C: Most red- \n shifted $v_{pri}$":	
					largestAs = pointspeakovernoise
					largestSigs = pointswidthofpeak
	
			
			ax.scatter(np.array(gammas)/5.,pointspeakovernoise,color=colors,marker=marker,edgecolors=edgecolors,s=markersize)
			if num == 0:		
				ax.set_ylabel("Gaussian Peak Height \n over Noise", fontsize=16)
		
			#if parameter == "PeakHeightatKp" :
			#	ax.scatter(gammas,pointsheightofpeak,color=colors)	
			#	if num == 0:		
			#		ax.set_ylabel("Gaussian Peak Height", fontsize=16)	
	
			#if parameter == "PeakWidthatKp":
			#	ax.scatter(gammas,pointswidthofpeak,color=colors,marker=marker,edgecolors=edgecolors,s=markersize)	
			#	if num == 0:		
			#		ax.set_ylabel("1 / (Gaussian Peak Width)", fontsize=16)	
		
		AllRsq.append(midrsq)
		
		ax.set_title(titles[num],fontsize=16)
		for tick in ax.yaxis.get_major_ticks(): tick.label.set_fontsize(14) 
		#for tick in ax.xaxis.get_major_ticks(): tick.label.set_fontsize(14)	

Panels = "bottom"
if Panels == "bottom":
	axes = bigaxes[1]
	parameter = "PeakWidthatKp"
	
	for num, ax in enumerate(axes):
		for kindofpoint in range(2):
			#points = []						###  height at 75 relative to std from -150 to 0
			pointspeakovernoise = []		### heigh of Gaussian within 1sigma of Kp over std from -150 to 0
			#pointsheightofpeak = []			###  height of Gaussian if there is one within 1sigma of Kp
			pointswidthofpeak = []			###  width of Gaussian if there is one within 1sigma of Kp
			gammas = []
			colors = []

			if kindofpoint == 0:
				versions = allversions[num]
				marker = 'o'
				edgecolors = 'none'
				markersize = None
			if kindofpoint == 1:
				versions = specialversions[num]
				marker = '*'
				edgecolors = 'k'
				markersize = 200
				
			for version in versions:
				version = int(version)
				
				try:
					totaldir = '/home/cbuzard/Pipeline/03_CrossCorr/Output/VpriM_simulations_version'
					Kps = 75
	
					if type == 'inv':
						dir = totaldir + str(version) + '/inv/COMBO_3/'
					if type == 'noninv':
						dir = totaldir + str(version) + '/noninv/COMBO_3/'
					m = readsav(dir+'max_like_vb_VpriM_simulations.idl.sav')
				except IOError:
					totaldir = '/export/nobackup1/cbuzard/Pipeline/03_CrossCorr/Vpri_Output/VpriM_simulations_version'
					Kps = 75
	
					if type == 'inv':
						dir = totaldir + str(version) + '/inv/COMBO_3/'
					if type == 'noninv':
						dir = totaldir + str(version) + '/noninv/COMBO_3/'
					m = readsav(dir+'max_like_vb_VpriM_simulations.idl.sav')
					
				x_all, y_all = m.toplotx, m.toploty
		
				#plt.figure(1)
				#plt.plot(x_all,y_all)
						
	
				#### measure gamma
				nightsdir = '/home/cbuzard/Pipeline/03_CrossCorr/Targets/VpriM_simulations/version'+str(version)+'_M_nights.dat'
				Ms = np.genfromtxt(nightsdir, skip_header=1)[0]
				gamma = np.sum(np.abs(np.sin(2*np.pi*Ms)))
				gammas.append(gamma)

				#if parameter == "PeakHeightOverNoise":
				#	#### height of point at Kp (75 km/s) over std from -150 to 0
				#	PeakHeightOverNoise(Kps,x_all,y_all,points)
				#
				#if parameter == "PeakHeightatKp" or "PeakWidthatKp":
				#	### if there is peak within 1sigma of 75, how tall is it
				PeakHeightatKp(Kps,x_all,y_all,pointswidthofpeak,pointspeakovernoise,colors,version, titles[num], gamma)
	
	
			#if parameter == "PeakHeightOverNoise":
			#	ax.scatter(gammas,pointspeakovernoise,color=colors,marker=marker,edgecolors=edgecolors,s=markersize)
			#	if num == 0:		
			#		ax.set_ylabel("Gaussian Peak Height \n over Noise", fontsize=16)
		
			#if parameter == "PeakHeightatKp" :
			#	ax.scatter(gammas,pointsheightofpeak,color=colors)	
			#	if num == 0:		
			#		ax.set_ylabel("Gaussian Peak Height", fontsize=16)	
	
			#if parameter == "PeakWidthatKp":
			ax.scatter(np.array(gammas)/5.,pointswidthofpeak,color=colors,marker=marker,edgecolors=edgecolors,s=markersize)	
			if num == 0:		
				ax.set_ylabel("Gaussian Peak Width [km/s]", fontsize=16)	
		
		DrawNIRSPECvelocityprecisionline(ax)
		
		#ax.set_title(titles[num],fontsize=16)
		for tick in ax.yaxis.get_major_ticks(): tick.label.set_fontsize(14) 
		for tick in ax.xaxis.get_major_ticks(): tick.label.set_fontsize(14)	

	
fig.text(0.5, -0.02, r"$\gamma = \Sigma |\sin (2\pi M)| / N$", ha='center', fontsize=16)

plt.tight_layout()	
plt.savefig("/home/cbuzard/Desktop/VpriPaper/gaussianfitparams_"+type+".pdf", bbox_inches='tight')
#plt.savefig("/home/cbuzard/Desktop/VpriPaper/gaussianfitparams_"+type+"_heightsonly.pdf", bbox_inches='tight')
#plt.show()
#plt.close()


### KS test results
print('')
statistic, pvalue = stats.ks_2samp(nearzeroAs,randomAs)
xsigma = ConvertTwoTailedPValuetoSigma(pvalue)
print("Near Zero vs Random, A :: ", pvalue, xsigma)
statistic, pvalue = stats.ks_2samp(nearzeroSigs,randomSigs)
xsigma = ConvertTwoTailedPValuetoSigma(pvalue)
print("Near Zero vs Random, sigma :: ", pvalue, xsigma)


print('')
statistic, pvalue = stats.ks_2samp(largestAs,smallestAs)
xsigma = ConvertTwoTailedPValuetoSigma(pvalue)
print("Largest vs Smallest, A :: ", pvalue, xsigma)
statistic, pvalue = stats.ks_2samp(largestSigs,smallestSigs)
xsigma = ConvertTwoTailedPValuetoSigma(pvalue)
print("Largest vs Smallest, sigma :: ", pvalue, xsigma)



## Rsq
print('')
print('Mean Gaussian fit R squared')
meanfitrsq = np.nanmean(AllRsq,axis=1)
stdfitrsq = np.nanstd(AllRsq,axis=1)
print("Smallest :: ", meanfitrsq[0],stdfitrsq[0])
print("Even :: ", meanfitrsq[1],stdfitrsq[1])
print("Largest :: ", meanfitrsq[2],stdfitrsq[2])
print("Near Zero :: ", meanfitrsq[3],stdfitrsq[3])
print("Random :: ", meanfitrsq[4],stdfitrsq[4])
print('')
	

assert(1==0)

assert(len(nearzeroAs)==100)
assert(len(randomAs)==100)
assert(len(nearzeroSigs)==100)
assert(len(randomSigs)==100)
print('Testing subgroups of 20 elements')
sigmadiffpopulationsforA = []
for ii in range(100):
	nearzeroAs_subgroup = np.array(nearzeroAs)[np.random.randint(0,100,size=20)]
	randomAs_subgroup = np.array(randomAs)[np.random.randint(0,100,size=20)]
	statistic, pvalue = stats.ks_2samp(nearzeroAs_subgroup,randomAs_subgroup)
	xsigma = ConvertTwoTailedPValuetoSigma(pvalue)
	sigmadiffpopulationsforA.append(xsigma)

print('')
print('')
print("The mean sigma level at which we can say random 20 element subgroups of the near zero and random A sets are different is")
print(str(np.mean(sigmadiffpopulationsforA)) +' +/- '+  str(np.std(sigmadiffpopulationsforA)))
print('')
print('compare to inf for full 100 element A sets')	
print('and compare to KS results from different order tests, which are about')
print('order 0 :: 3.3798378601679744')
print('order 1 :: 2.2699061836677505')
print('order 2 :: 1.49487469315515')


