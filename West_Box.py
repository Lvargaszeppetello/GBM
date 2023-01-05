import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rand
from netCDF4 import Dataset

global steps_per_day
steps_per_day = 20

def write_time_series(SET_NAME,var_name,ARRAY):
	new_set = Dataset(SET_NAME,'w',format='NETCDF3_64BIT')
	new_set.createDimension('time',len(ARRAY))
	MAIN_VAR = new_set.createVariable(var_name,'f4',('time',))
	MAIN_VAR[:] = ARRAY
	new_set.close()

def sumP(P,Nyears):
	
	mon_len = np.array([0,31,28,31,30,31,30,31,31,30,31,30,31])
	mon_len = np.cumsum(mon_len)
	Pmon = np.zeros(Nyears*12)
	i = 0
	while i < Nyears:
		j = 0	
		while j < 12:
			Pmon[i*12 + j] = np.sum(P[(365*i + mon_len[j])*steps_per_day:(365*i + mon_len[j+1])*steps_per_day])
			j+=1
		i+=1

	return(Pmon)

def e_s(TC):
	
	# Function to calculate saturation mixing ratio. Returns pressure in hPa
	e_s = 6.11*10**(7.5*TC/(237.5+TC))  # TEMPERATURES IN CELSIUS!!!
	return(e_s)

def calc_q_s(TC,P):	
	# Function to calculate saturation mixing ratio as a function of temperature (in celsius) and pressure in hPa
	es = e_s(TC)
	return(es*0.622/(P - 0.37*es))

def calc_delta(TC,P):

	TC_plus = TC+0.05
	TC_minus = TC-0.05
	delta = (calc_q_s(TC_plus,P) - calc_q_s(TC_minus,P))/(TC_plus - TC_minus)
	return(delta)

def make_forcing(Nyears):

	#### RETURN F_forcing, T_forcing, Q_forcing

	days_per_year = 365

	Nmons = Nyears*12

	N = Nyears*days_per_year*steps_per_day

	Time = np.arange(Nmons)
	Time_model = np.linspace(0,Nmons-1,N)

	Fmean = 160
	Famp = 80
	rand_F = 20

	F_cyc = -Famp*np.cos(2*np.pi*Time/12)

	T_mean = 280
	T_amp = 4
	rand_T = 0
	T_cyc = -T_amp*np.cos(2*np.pi*Time/12 - np.pi/6)

	q_mean = 0.003
	q_amp  = 0
	rand_q = 0

	q_cyc = -q_amp*np.cos(2*np.pi*Time/12 - np.pi/6)	

	i = 0
	qnoise = np.zeros(Nmons)
	Fnoise = np.zeros(Nmons)
	Tnoise = np.zeros(Nmons)

	qwalk = rand.normal(0,rand_T,size=Nmons)*rand_q
	Fwalk = rand.normal(0,rand_F,size=Nmons)
	Twalk = rand.normal(0,rand_T,size=Nmons)

	r = 0.15
	while i < Nmons-1:
		qnoise[i+1] = r*qnoise[i] + qwalk[i]
		Fnoise[i+1] = r*Fnoise[i] + Fwalk[i]
		Tnoise[i+1] = r*Tnoise[i] + Twalk[i]

		i+=1

	q_forcing = q_mean + q_cyc + qnoise
	F_forcing = Fmean + F_cyc + Fnoise
	T_forcing = T_mean + T_cyc + Tnoise
	qs = calc_q_s(T_forcing-273.15,900)

	RH_forcing = q_forcing/calc_q_s(T_forcing-273.15,900)

####################### Making a supplemental Figure

#	plt.figure(figsize=(15,6))
#	plt.subplot(3,1,1)
#	plt.plot(np.arange(120),F_forcing[:120],'k')
#	plt.subplot(3,1,2)
#	plt.plot(np.arange(120),T_forcing[:120],'k')
#	plt.subplot(3,1,3)
#	plt.plot(np.arange(120),RH_forcing[:120],'k')
#	plt.plot(np.arange(120),qs[:120])
#	plt.plot(np.arange(120),q_forcing[:120],'k')
#	plt.savefig('SI_Forcing.pdf')
#	plt.show()
#	f = breakhere

################################################################

	#### GETTING RARE CASES OF NEGATIVE NET DOWNWARD SOLAR
	np.putmask(F_forcing,F_forcing<0,0)

	F_forcing = np.interp(Time_model,Time,F_forcing)
	q_forcing = np.interp(Time_model,Time,q_forcing)
	T_forcing = np.interp(Time_model,Time,T_forcing)

	return(F_forcing,T_forcing,q_forcing)

def West_Box(F,T_R,q_R,coupled):

##################################################################################
###################### SIMPLE MODEL FOR FIGURE 3 #########################
##################################################################################

	from scipy.stats import pearsonr

################################ PHYSICAL CONSTANTS ###############################

	N = len(F)	 		# Number of soil moisture values we simulate
	rho_a = 1.25 		# [kg/m^3]
	c_p   = 1003		# [J/kg/K]	
	c_s   = 1000		# [J/kg/K]
	rho_s = 1000		# [kg/m^3]
	L     = 2257000 	# [J/kg]
	T_freeze = 273.15	# Kelvin
	P_s	 = 900		# hPa
	rho_l = 1000		# density of water	

############################# PARAMETERS ##########################################
	
	theta_max = 0.5		# soil pore space [-]
	h 	= 1000		# meters
	h_s 	= .1		# meter
	g_o	= 1/500. 	# [m/s]
	nu_BL_T = 20
	nu_H 	= 20		# Dry surface sensitivity
	nu_BL_q = rho_a*h/(1*86400.)

############## PRECIP STUFF ###################################################

	P_avg = 8				# average precipitation intensity [mm]
	a_1 = 2				# precip frequency [days]
	alpha = a_1/steps_per_day
####################################################################################

	RH_F = q_R/calc_q_s(T_R-T_freeze,P_s)
	min_RHF = np.nanmin(RH_F)

	if coupled == False:
		uncoup = 1		
		coup = 0
		beta = 0.27
	if coupled == True:
		uncoup = 0
		coup = 1
		beta = 0.05

	sec_per_day = 86400		# seconds per day
	dt = 86400./steps_per_day		# time increment (10 chunks per day)
	i = 0

	theta = np.zeros(N)
	Ts = np.zeros(N)
	q = np.zeros(N)
	m = np.zeros(N)
	LHF = np.zeros(N)
	P = np.zeros(N)
	RH = np.zeros(N)

	theta[0] = T_R[0]
	Ts[0] = T_R[0]
	q[0] = q_R[0]
	m[0] = 0

	while i < N-1:

		H = nu_H*(Ts[i] - theta[i])
		q_s = calc_q_s(Ts[i]-T_freeze,P_s)			# Saturation Specific Humidity
		q_def = q_s - q[i]					# Specific Humidity Gradient
		g_s = m[i]*g_o						# Surface Conductance 

		if q_def > 0:
			ET = rho_a*g_s*q_def				# Evapotranspiration
		else:
			ET = 0

		LHF[i] = L*ET

		H_R = nu_BL_T*(theta[i] - T_R[i])			# Reference Energy Flux
		Q_R = nu_BL_q*(q[i] - q_R[i])				# Reference Moisture Flux

		RH[i] = q[i]/q_s
		does_rain = rand.rand()

		omega = (uncoup*alpha*(RH_F[i] - beta)) + (coup*alpha*(RH[i] - beta))

		if does_rain < omega:
			P[i] = rand.gamma(P_avg,scale=1)
	
		dT_dt = (F[i] - H - LHF[i])/(c_s*rho_s*h_s)
		dtheta_dt = (H - H_R)/(c_p*rho_a*h)
		dq_dt = (ET - Q_R)/(rho_a*h)
		dm_dt = (-ET)/(rho_l*h_s*theta_max)

		Ts[i+1] = Ts[i] + dT_dt*dt
		theta[i+1] = theta[i] + dtheta_dt*dt
		q[i+1] = q[i] + dq_dt*dt
		m[i+1] = m[i] + dm_dt*dt + (1-(m[i]/theta_max))*P[i]/(1000*h_s*theta_max)

		if m[i+1] > 1:
			m[i+1] = 1
		if m[i+1] < 0:
			m[i+1] = 0
		i+=1

	return(Ts,theta,m*theta_max,q,P)
	

#def make_contours(mam_m,jja_T):
def make_contours():

	from scipy.stats import pearsonr
	Tset = Dataset('LA_JJA_T.nc','r')
	mset = Dataset('LA_MAM_m.nc','r')
	
#	Tset = Dataset('FORCED_JJA_T.nc','r')
#	mset = Dataset('FORCED_MAM_m.nc','r')

	mam_m = mset['m'][:]
	jja_T = Tset['T'][:]
	print(np.nanmax(mam_m))

	data = []
	mylen = []
	positions = []
	quantiles = []
	bounds = [-0.06,-0.05,-0.04,-0.03,-0.02,-.01,0.01,0.02,0.03,0.04,0.05,0.06]

	i = 0
	while i < len(bounds) - 1:
		mybin = jja_T[np.logical_and(mam_m<bounds[i+1],mam_m>bounds[i])]
		if len(mybin)>10:
			mylen.append(len(mybin)/1000.)
			data.append(mybin)
			positions.append((bounds[i+1]*50)+bounds[i]*50)
			quantiles.append([.25,.75])
		i+=1

#	positions = np.array([-55,-45,-34,-25,-15,0,15,25,35,45,55])/10.
#	fig = plt.figure(figsize=(10,6))
#	ax = fig.add_axes([0,0,1,1])
#	bp = ax.boxplot(data)
#	plt.boxplot(data,
#		positions=positions,whis=[30,60],
#			widths=mylen,showfliers=False)

	plt.figure(figsize=(4,10))
	plt.violinplot(data,positions=positions,widths=mylen,quantiles=quantiles,showextrema=False,showmedians=True)

	p = np.polyfit(mam_m,jja_T,deg = 1)
	print(p)
	print(pearsonr(mam_m,jja_T))
	plt.yticks([6,7,8,9,10,11,12,13,14],[6,7,8,9,10,11,12,13,14])
	X = np.asarray(positions)
	plt.plot(X,(X*p[0]/100. + p[1]),'k--',linewidth=3)

	plt.ylim(-2,2)
	plt.yscale('linear')
#	plt.yscale('log')
#	print(pearsonr(mam_m,jja_T))
	
	plt.savefig('LA_violin.pdf')
	plt.show()

#	H,xedges,yedges = np.histogram2d(mam_m,jja_T,bins=20,range=[[-.08,.08],[-1.6,1.6]])
#	xbins = (xedges[1:] + xedges[:-1])/2.
#	ybins = (yedges[1:] + yedges[:-1])/2.
#
#	p = np.polyfit(mam_m,jja_T,deg = 1)
#
#	print(p)
#	print(pearsonr(mam_m,jja_t))
#
#	X = np.linspace(-0.1,0.1,10)
#	plt.contour(xbins,ybins,H,cmap='ocean_r',linewidths=3)
#	plt.plot(X,X*p[0] + p[1],'k--',linewidth=3)
#	plt.ylim(-1.7,1.7)
#	plt.xlim(-.08,0.08)
#	plt.savefig('LA_contour.pdf')
#	plt.show()

def Seasonal_Cyc():

	Nyears = 5000
	f,t,q = make_forcing(Nyears)

	ts_no,theta_no,m_no,q_no,p_no = West_Box(f,t,q,False)
	ts_yes,theta_yes,m_yes,q_yes,p_yes = West_Box(f,t,q,True)

	months = np.linspace(0,Nyears*12 -1,Nyears*12)
	Time = np.linspace(0,Nyears*12-1,steps_per_day*Nyears*365)

#	fmon = np.interp(months,Time,f)
#	fmon = np.reshape(fmon,(Nyears,12))
#	fcyc = np.nanmean(fmon,axis=0)
#
	Utheta_month = np.interp(months,Time,theta_no)
	Um_month = np.interp(months,Time,m_no)
	Uq_month = np.interp(months,Time,q_no)
	UP_month = sumP(p_no,Nyears)

	Um_month = np.reshape(Um_month,(Nyears,12))
	Utheta_month = np.reshape(Utheta_month,(Nyears,12))
	Uq_month = np.reshape(Uq_month,(Nyears,12))
	UP_month = np.reshape(UP_month,(Nyears,12))

	Utheta_cyc = np.nanmean(Utheta_month,axis=0)
	Uq_cyc = np.nanmean(Uq_month,axis=0)
	Um_cyc = np.nanmean(Um_month,axis=0)
	UP_cyc = np.nanmean(UP_month,axis=0)

	Ctheta_month = np.interp(months,Time,theta_yes)
	Cm_month = np.interp(months,Time,m_yes)
	Cq_month = np.interp(months,Time,q_yes)
	CP_month = sumP(p_yes,Nyears)

	Cm_month = np.reshape(Cm_month,(Nyears,12))
	Ctheta_month = np.reshape(Ctheta_month,(Nyears,12))
	Cq_month = np.reshape(Cq_month,(Nyears,12))
	CP_month = np.reshape(CP_month,(Nyears,12))

#	return(Um_month,Utheta_month)

	Ctheta_cyc = np.nanmean(Ctheta_month,axis=0)
	Cq_cyc = np.nanmean(Cq_month,axis=0)
	Cm_cyc = np.nanmean(Cm_month,axis=0)
	CP_cyc = np.nanmean(CP_month,axis=0)

	URH_cyc = Uq_cyc/calc_q_s(Utheta_cyc-273.15,900)
	CRH_cyc = Cq_cyc/calc_q_s(Ctheta_cyc-273.15,900)

######################## SUPPLEMENTAL FIGURE
	'''
	mons = np.linspace(1,12,12)

	plt.figure(figsize=(15,8))
	ax0 = plt.subplot(511)
	ax1 = ax0.twinx()
	ax2 = plt.subplot(512)
	ax3 = ax2.twinx()
	ax4 = plt.subplot(513)
	ax5 = ax4.twinx()
	ax6 = plt.subplot(514)
	ax7 = ax6.twinx()
	ax8 = plt.subplot(515)
	ax9 = ax8.twinx()

	ax0.set_xticks([])
	ax1.set_xticks([])
	ax2.set_xticks([])
	ax3.set_xticks([])
	ax4.set_xticks([])
	ax5.set_xticks([])
	ax6.set_xticks([])
	ax7.set_xticks([])

	ax0.plot(mons,Utheta_cyc,'k',linewidth=2)
	ax0.plot(mons,Ctheta_cyc,'k--',linewidth=2)
	ax1.plot(mons,Utheta_cyc - Ctheta_cyc,'r',linewidth=2)
	ax1.set_ylim(-1,1)
	ax1.tick_params(axis='y', labelcolor='r')


	ax2.plot(mons,100*URH_cyc,'k',linewidth=2)
	ax2.plot(mons,100*CRH_cyc,'k--',linewidth=2)
	ax3.plot(mons,(URH_cyc - CRH_cyc)*100,'r',linewidth=2)
	ax3.set_ylim(-3,3)
	ax3.tick_params(axis='y', labelcolor='r')
	


	ax4.plot(mons,UP_cyc,'k',linewidth=2)
	ax4.plot(mons,CP_cyc,'k--',linewidth=2)
	ax5.plot(mons,UP_cyc - CP_cyc,'r',linewidth=2)
	ax5.set_ylim(-50,50)
	ax5.tick_params(axis='y', labelcolor='r')


	ax6.plot(mons,Uq_cyc*1000,'k',linewidth=2)
	ax6.plot(mons,Cq_cyc*1000,'k--',linewidth=2)
	ax7.plot(mons,(Uq_cyc - Cq_cyc)*1000,'r',linewidth=2)
	ax7.set_ylim(-.5,.5)
	ax7.tick_params(axis='y',labelcolor='r')


	ax8.plot(mons,Um_cyc,'k',linewidth=2)
	ax8.plot(mons,Cm_cyc,'k--',linewidth=2)
	ax9.plot(mons,Um_cyc - Cm_cyc,'r',linewidth=2)
	ax9.set_ylim(-.1,.1)
	ax9.tick_params(axis='y',labelcolor='r')

	plt.savefig('SI_Seasonal_Cycle.pdf')
	plt.show()
	'''

#	return(Cm_month,Ctheta_month)
	return(Um_month,Cm_month,Utheta_month,Ctheta_month)

def Big_Contour():
	um,cm,ut,ct = Seasonal_Cyc()
	um_mam = np.nanmean(um[10:,2:5],axis=1)
	uT_jja = np.nanmean(ut[10:,5:8],axis=1)

	cm_mam = np.nanmean(cm[10:,2:5],axis=1)
	cT_jja = np.nanmean(ct[10:,5:8],axis=1)

	um_anoms = um_mam - np.nanmean(um_mam,axis=0)
	uT_anoms = uT_jja - np.nanmean(uT_jja,axis=0)

	cm_anoms = cm_mam - np.nanmean(cm_mam,axis=0)
	cT_anoms = cT_jja - np.nanmean(cT_jja,axis=0)

	write_time_series('NOISY_FORCED_JJA_T.nc','T',uT_anoms)
	write_time_series('NOISY_FORCED_MAM_m.nc','m',um_anoms)

	write_time_series('NOISY_LA_JJA_T.nc','T',cT_anoms)
	write_time_series('NOISY_LA_MAM_m.nc','m',cm_anoms)


######### MAKE FORCING FIGURE ########################
#make_forcing(20)
#############################333#######################

############# MAKE VIOLIN PLOT #################333
make_contours()
f = breakhere
###################################################

