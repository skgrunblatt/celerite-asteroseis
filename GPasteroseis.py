import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage

import celerite
from celerite import terms
from celerite.modeling import Model, ConstantModel
import muhz2days

from matplotlib import rcParams
rcParams["savefig.dpi"] = 200

#Read in lightcurve data
data = np.loadtxt('/Users/samuelgrunblatt/Desktop/reinflationpaper/EPIC228754001/228754001.dat.ts')

t = data[:,0]
y = (data[:,1]-1) #avg=0, not 1
yerr = np.median(np.abs(np.diff(y)))

muhzconv = 1e6 / (3600*24)

def muhz2idays(muhz):
    return muhz / muhzconv

def muhz2omega(muhz):
    return muhz2idays(muhz) * 2.0 * np.pi

def idays2muhz(idays):
    return idays * muhzconv

def omega2muhz(omega):
    return idays2muhz(omega / (2.0 * np.pi))

omega = np.linspace(1, 300, 1000)
freqmuhz = omega2muhz(omega)
freqidays = muhz2idays(freqmuhz)

#sigma clip and smooth if necessary
print("start lengths:", len(t), len(y))
#sigmaclip and remove nans
sigclip = 9.5
loopsize = 10000
oldlentimes = len(t)+1
while oldlentimes > len(t):
    newtimes = np.array([])
    newfluxes = np.array([])
    j=0
    print(len(t) / loopsize + 1)
    while j < len(t) / loopsize + 1:
        dellist = []
        piecet = t[loopsize*j:loopsize*j+loopsize]
        piece = y[loopsize*j:loopsize*j+loopsize]
        for k in range(len(piece)):
            if abs(piece[k] - np.nanmean(piece)) > sigclip*np.nanstd(piece) or np.isnan(piece[k]) == True:
                dellist.append(k)
        piecet = np.delete(piecet,dellist)
        piece = np.delete(piece,dellist)
        newtimes = np.append(newtimes, piecet)
        newfluxes = np.append(newfluxes, piece)
        j=j+1
    print("new lens:", len(newtimes), len(newfluxes))
    oldlentimes = len(t)
    t = newtimes
    y = newfluxes
print("finish lengths:", len(t), len(y))

#median filter and remove outliers
y -= scipy.ndimage.filters.median_filter(y, size=150) #50 datapts ~ 1 day

#2287: remove edge effect during midcampaign break
edgemask = (t > 6.5) * (t < 22.5)
t = t[-edgemask]
y = y[-edgemask]

from ktransit import FitTransit
import ktransit


fitT = FitTransit()

#EPIC2287
fitT.add_guess_star(rho=0.0264, zpt=0, ld1=0.6505,ld2=0.1041) 
fitT.add_guess_planet(T0=7.5, period=9.17, impact=0.8, rprs=0.03, ecosw=0.0, esinw=0.0) #simult, sff

fitT.add_data(time=t,flux=np.array(y))

vary_star = ['zpt']#, 'ld1', 'ld2']      # free stellar parameters
vary_planet = (['period',       # free planetary parameters
        'T0', 'impact',# 'esinw', 'ecosw',
        'rprs'])                # free planet parameters are the same for every planet you model

fitT.free_parameters(vary_star, vary_planet)
fitT.do_fit()                   # run the fitting

fitT.print_results()            # print some results
res=fitT.fitresultplanets
res2=fitT.fitresultstellar

print("transit model shape:", fitT.transitmodel.shape)

class TransitModel(Model):
    parameter_names = ("log_ror", "log_rho", "log_T0", "log_per", "log_imp")#, "ecosw", "esinw")
    
    def get_value(self, t):
        
         #make transit model
        M=ktransit.LCModel()
        M.add_star(
            rho=np.exp(self.log_rho), # mean stellar density in cgs units
            ld1=0.6505, # ld1--4 are limb darkening coefficients 
            ld2=0.1041, # if only ld1 and ld2 are non-zero then a quadratic limb darkening law is used
            #ld1=theta[8],
            #ld2=theta[9],
            ld3=0.0, # if all four parameters are non-zero we use non-linear flavour limb darkening
            ld4=0.0, 
            dil=0.0, # a dilution factor: 0.0 -> transit not diluted, 0.5 -> transit 50% diluted
            zpt=0.0  # a photometric zeropoint, incase the normalisation was wonky
            )
        M.add_planet(
            T0=np.exp(self.log_T0),     # a transit mid-time  
            period=np.exp(self.log_per), # an orbital period in days
            impact=np.exp(self.log_imp), # an impact parameter
            rprs=np.exp(self.log_ror),   # planet stellar radius ratio  
            #ecosw=self.ecosw,  # eccentricity vector
            #esinw=self.esinw,
            occ=0.0)    # a secondary eclipse depth in ppm

    
        M.add_data(time=t)

        return M.transitmodel

#set the GP parameters
Q = 1.0 / np.sqrt(2.0)
w0 = muhz2omega(13)
S0 = np.var(y) / (w0*Q)
kernel = terms.SHOTerm(log_S0=np.log(S0), log_Q=np.log(Q), log_omega0=np.log(w0),
                       bounds=[(-25, 0), (-15, 15), (np.log(muhz2omega(3)), np.log(muhz2omega(50)))]) #omega upper bound: 275 muhz
kernel.freeze_parameter("log_Q") #to make it a Harvey model

Q = 1.0 / np.sqrt(2.0)
w0 = muhz2omega(61.0)
S0 = np.var(y) / (w0*Q)
kernel += terms.SHOTerm(log_S0=np.log(S0), log_Q=np.log(Q), log_omega0=np.log(w0),
                   bounds=[(-25, 0), (-15, 15), (np.log(muhz2omega(30)), np.log(muhz2omega(1000)))])
kernel.freeze_parameter("terms[1]:log_Q") #to make it a Harvey model

Q = np.exp(3.0)
w0 = muhz2omega(250) #peak of oscillations at 220 muhz
S0 = np.var(y) / (w0*Q)
kernel += terms.SHOTerm(log_S0=np.log(S0), log_Q=np.log(Q), log_omega0=np.log(w0),
                      bounds=[(-40, 0), (0.5, 4.2), (np.log(muhz2omega(200)), np.log(muhz2omega(280)))])

mean = TransitModel(log_ror=np.log(res['pnum0']['rprs']), log_rho=np.log(0.008),
                    log_T0=np.log(res['pnum0']['T0']), log_per=np.log(res['pnum0']['period']),
                    log_imp=np.log(res['pnum0']['impact']), 
                    bounds=[(-5, 0), (-6.0,-3.0), (np.log(res['pnum0']['T0']-1), np.log(res['pnum0']['period'])), (1,3),(np.log(0.01),np.log(1.1))])#, ((np.log(res['pnum0']['T0']-5),
                    #np.log(res['pnum0']['T0']+5))-1,1), (-1,1)]) # ecosw=res['pnum0']['ecosw'], esinw=res['pnum0']['esinw'], 
#mean.freeze_parameter("log_rho")
mean.freeze_parameter("log_per")
#mean.freeze_parameter("log_T0")
#mean.freeze_parameter("log_imp")

kernel += terms.JitterTerm(log_sigma=-10, bounds=[(-20,20)])

gp = celerite.GP(kernel, mean=mean, fit_mean=True)#, log_white_noise=np.log(np.mean(yerr)**2/len(t)), fit_white_noise=False)
gp.compute(t, yerr)

#find max likelihood params
from scipy.optimize import minimize

def neg_log_like(params, y, gp):
    gp.set_parameter_vector(params)
    ll = gp.log_likelihood(y)
    if not np.isfinite(ll):
        return 1e10
    return -ll

initial_params = gp.get_parameter_vector()
bounds = gp.get_parameter_bounds()

r = minimize(neg_log_like, initial_params, method="L-BFGS-B", bounds=bounds, args=(y, gp))
gp.set_parameter_vector(r.x)
print(r)

#create the power spectral density

from astropy.stats import LombScargle

#model = LombScargle(t, y*1e-3)
model = LombScargle(t, y)
power_ls = model.power(freqidays, method="fast", normalization="psd")
power_ls /= len(t)


#make LC, SC powspec
import lomb

osample=10.
nyq=283.
#nyq = 550

freq, amp, nout, jmax, prob = lomb.fasper(t,y, osample, 3.)
freq = 1000.*freq/86.4
binn = freq[1]-freq[0]
fts = 2.*amp*np.var(y)/(np.sum(amp)*binn)
fts = scipy.ndimage.filters.gaussian_filter(fts, 4)
use=np.where(freq < nyq+150)
freq=freq[use]
print(len(freq))
fts=fts[use]

#set up MCMC
rhotrue= 0.0264
rhosigma = 0.0008

def lnprob(params, y, gp):
    gp.set_parameter_vector(params)
    lp = gp.log_prior()
    if not np.isfinite(lp):
        return -np.inf
    
    ll = gp.log_likelihood(y)
    
    if not np.isfinite(ll):
        return -np.inf
    return ll + lp - (((rhotrue - (np.e**params[9]))/rhosigma)**2/2.) #+ np.log(1./np.sqrt(params[-2]**2 + params[-1]**2))

# Set up the sampler.
import emcee
#import triangleedit
import time

merged_par = gp.get_parameter_vector()

from matplotlib.ticker import MaxNLocator
ndim, nwalkers = len(merged_par), 30
pos = [merged_par *(1+ 5e-4*np.random.randn(ndim)) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(y,gp), threads=8) 
    
# Time an MCMC step, and estimate length of time to run chain.
t0 = time.time()
sampler.run_mcmc(pos, 1, rstate0=np.random.get_state())
tstep = time.time()-t0
Nsteps = 50000
from time import localtime, strftime
tstart = strftime("%a, %d %b %Y %H:%M:%S", localtime())
print("""MCMC step runtime (in seconds): {0}
        MCMC estimated total runtime (in seconds): {1}
        Start time: {2}
        """.format(tstep, tstep * Nsteps, tstart))

# Clear and run the production chain.
print("Running MCMC...")
sampler.run_mcmc(pos, Nsteps, rstate0=np.random.get_state())
print("Done.")
print("End time:", strftime("%a, %d %b %Y %H:%M:%S", localtime()))

burnin = 5000
samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))

lnrprssamples = sampler.chain[:, burnin:, 8].flatten()


[log_s00_mcmc, log_omega00_mcmc, log_s01_mcmc, log_omega01_mcmc, log_s02_mcmc, \
    log_q2_mcmc, log_omega02_mcmc, log_sigma_mcmc, log_rprs_mcmc, log_rho_mcmc, \
 log_T0_mcmc, log_imp_mcmc] = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                            zip(*np.percentile(samples, [16, 50, 84], axis=0))) 
print("""MCMC result:
lns00 = {0[0]} +{0[1]} -{0[2]} 
lnomega00 = {1[0]} +{1[1]} -{1[2]}
lns01 = {2[0]} +{2[1]} -{2[2]} 
lnq1 = {3[0]} +{3[1]} -{3[2]}
lns02 = {4[0]} +{4[1]} -{4[2]} 
lnq2 = {5[0]} +{5[1]} -{5[2]}
log(muhztoomega(numax)) = {6[0]} +{6[1]} -{6[2]}
lnsigma = {7[0]} +{7[1]} -{7[2]} 
lnrprs = {8[0]} +{8[1]} -{8[2]}
lnrho = {9[0]} +{9[1]} -{9[2]}
lnT0 = {10[0]} +{10[1]} -{10[2]}
lnimp = {11[0]} +{11[1]} -{11[2]}
""".format(log_s00_mcmc, log_omega00_mcmc, log_s01_mcmc, log_omega01_mcmc, log_s02_mcmc, \
    log_q2_mcmc, log_omega02_mcmc, log_sigma_mcmc, log_rprs_mcmc, log_rho_mcmc, log_T0_mcmc, \
    log_imp_mcmc))

print("Mean acceptance fraction: {0:.3f}"
        .format(np.mean(sampler.acceptance_fraction)))

print('rprs:',np.exp(log_rprs_mcmc[0]),' +', np.exp(log_rprs_mcmc[0]+log_rprs_mcmc[1])-np.exp(log_rprs_mcmc[0]), \
      ' -', np.exp(log_rprs_mcmc[0]+log_rprs_mcmc[2])+np.exp(log_rprs_mcmc[0]))

print('numax:',omega2muhz(np.exp(log_omega02_mcmc[0])),' +', omega2muhz(np.exp(log_omega02_mcmc[1])), ' -', omega2muhz(np.exp(log_omega02_mcmc[2])))

#2287
Teff = np.sqrt(60**2+59**2)*np.random.randn(len(lnrprssamples)) + 4840. #add errors of Torres et al in quadrature
FeH = np.sqrt(0.04**2+0.062**2)*np.random.randn(len(lnrprssamples)) - 0.01
numax = 3.3*np.random.randn(len(lnrprssamples)) + 245.65 #inflate errors?
dnu = 0.26*np.random.randn(len(lnrprssamples)) + 18.48
fdnu = 0.998

Teffsun = 5777
numaxsun = 3100
dnusun = 135

Mstar = np.zeros_like(Teff)
Rstar = np.zeros_like(Teff)
rhostar = np.zeros_like(Teff)
logg = np.zeros_like(Teff)
Rp = np.zeros_like(lnrprssamples)

for i in range (len(Teff)):
    Mstar[i] = (numax[i] / numaxsun)**3 * (dnu[i] / (fdnu*dnusun))**(-4) * (Teff[i] / Teffsun)**(3/2)
    Rstar[i] = (numax[i] / numaxsun) * (dnu[i] / (fdnu*dnusun))**(-2) * (Teff[i] / Teffsun)**(1/2)
    rhostar[i] = 1.41 * (Mstar[i] / Rstar[i]**3)
    logg[i] = 4.43812 + np.log10(Mstar[i] / Rstar[i]**2)


    
#then use this to calculate planet radius with errors
for i in range(len(lnrprssamples)):
    Rp[i] = np.exp(lnrprssamples[i]) * Rstar[i] * 6.95e5 / 71492

print('Teff: ', np.percentile(Teff, 50), ' + ', np.percentile(Teff, 84.1)- np.percentile(Teff,50), ' - ', np.percentile(Teff,50) - np.percentile(Teff,14.9))
print('FeH: ', np.percentile(FeH, 50), ' + ', np.percentile(FeH, 84.1)- np.percentile(FeH,50), ' - ', np.percentile(FeH,50) - np.percentile(FeH,14.9))
print('Mstar: ', np.percentile(Mstar, 50), ' + ', np.percentile(Mstar, 84.1)- np.percentile(Mstar,50), ' - ', np.percentile(Mstar,50) - np.percentile(Mstar,14.9))
print('Rstar: ', np.percentile(Rstar, 50), ' + ', np.percentile(Rstar, 84.1)- np.percentile(Rstar,50), ' - ', np.percentile(Rstar,50) - np.percentile(Rstar,14.9))
print('rhostar: ', np.percentile(rhostar, 50), ' + ', np.percentile(rhostar, 84.1)- np.percentile(rhostar,50), ' - ', np.percentile(rhostar,50) - np.percentile(rhostar,14.9))
print('logg: ', np.percentile(logg, 50), ' + ', np.percentile(logg, 84.1)- np.percentile(logg,50), ' - ', np.percentile(logg,50) - np.percentile(logg,14.9))
print('Rp: ', np.percentile(Rp, 50), ' + ', np.percentile(Rp, 84.1)- np.percentile(Rp,50), ' - ', np.percentile(Rp,50) - np.percentile(Rp,14.9))

plt.clf()
fig, axes = plt.subplots(len(merged_par), 1, sharex=True, figsize=(8, 9))
axes[0].plot((sampler.chain[:, :, 0].T), color="k", alpha=0.4)
axes[0].yaxis.set_major_locator(MaxNLocator(5))
axes[0].axhline(np.median(sampler.chain[:, :, 0].T), color="#888888", lw=2)
axes[0].set_ylabel(r"ln$S_{0,0}$")
    
axes[1].plot((sampler.chain[:, :, 1].T), color="k", alpha=0.4)
axes[1].yaxis.set_major_locator(MaxNLocator(5))
axes[1].axhline(np.median(sampler.chain[:, :, 1].T), color="#888888", lw=2)
axes[1].set_ylabel(r"ln$\omega_{0,0}$")

axes[2].plot((sampler.chain[:, :, 2].T), color="k", alpha=0.4)
axes[2].yaxis.set_major_locator(MaxNLocator(5))
axes[2].axhline(np.median(sampler.chain[:, :, 2].T), color="#888888", lw=2)
axes[2].set_ylabel(r"ln$S_{0,1}$")
    
axes[3].plot((sampler.chain[:, :, 3].T), color="k", alpha=0.4)
axes[3].yaxis.set_major_locator(MaxNLocator(5))
axes[3].axhline(np.median(sampler.chain[:, :, 3].T), color="#888888", lw=2)
axes[3].set_ylabel(r"ln$\omega_{0,1}$")

axes[4].plot((sampler.chain[:, :, 4].T), color="k", alpha=0.4)
axes[4].yaxis.set_major_locator(MaxNLocator(5))
axes[4].axhline(np.median(sampler.chain[:, :, 4].T), color="#888888", lw=2)
axes[4].set_ylabel(r"ln$S_{0,2}$")

axes[5].plot((sampler.chain[:, :, 5].T), color="k", alpha=0.4)
axes[5].yaxis.set_major_locator(MaxNLocator(5))
axes[5].axhline(np.median(sampler.chain[:, :, 5].T), color="#888888", lw=2)
axes[5].set_ylabel(r"ln$Q_2$")

axes[6].plot((sampler.chain[:, :, 6].T), color="k", alpha=0.4)
axes[6].yaxis.set_major_locator(MaxNLocator(5))
axes[6].axhline(np.median(sampler.chain[:, :, 6].T), color="#888888", lw=2)
axes[6].set_ylabel(r"ln$\omega_{0,2}$")

axes[7].plot((sampler.chain[:, :, 7].T), color="k", alpha=0.4)
axes[7].yaxis.set_major_locator(MaxNLocator(5))
axes[7].axhline(np.median(sampler.chain[:, :, 7].T), color="#888888", lw=2)
axes[7].set_ylabel(r"ln$\sigma$")

axes[8].plot((sampler.chain[:, :, 8].T), color="k", alpha=0.4)
axes[8].yaxis.set_major_locator(MaxNLocator(5))
axes[8].axhline(np.median(sampler.chain[:, :, 8].T), color="#888888", lw=2)
axes[8].set_ylabel(r"ln$r_p/r_*$")

axes[9].plot((sampler.chain[:, :, 9].T), color="k", alpha=0.4)
axes[9].yaxis.set_major_locator(MaxNLocator(5))
axes[9].axhline(np.median(sampler.chain[:, :, 9].T), color="#888888", lw=2)
axes[9].set_ylabel(r"ln$\rho$")

axes[10].plot((sampler.chain[:, :, 10].T), color="k", alpha=0.4)
axes[10].yaxis.set_major_locator(MaxNLocator(5))
axes[10].axhline(np.median(sampler.chain[:, :, 10].T), color="#888888", lw=2)
axes[10].set_ylabel(r"lnT$_0$")

axes[11].plot((sampler.chain[:, :, 11].T), color="k", alpha=0.4)
axes[11].yaxis.set_major_locator(MaxNLocator(5))
axes[11].axhline(np.median(sampler.chain[:, :, 11].T), color="#888888", lw=2)
axes[11].set_ylabel(r"ln$b$")

fig.tight_layout(h_pad=0.0)
fig.show()

import corner
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


majorFormatter = FormatStrFormatter('%.2f')
fig, ax = plt.subplots()
ax.xaxis.set_major_formatter(majorFormatter)

plt.clf()
plt.rcParams['font.size']=15
plt.rcParams['savefig.dpi']=200
#, r"$e$cos$\omega$", r"$e$sin$\omega$"
fig = corner.corner(samples, labels=[r"ln$S0$", r"ln$\omega_0$", r"ln$S1$", r"ln$\omega_1$", r"ln$S2$", r"ln$Q2$", r"ln$\omega_2$",\
                                     r"ln$\sigma$", r"ln$r_p/r_*$",r"ln$\rho$", r"lnT$_0$", r"ln$b$"], quantiles=[0.16, 0.5, 0.84],
                      truths=[np.median(sampler.chain[:, :, 0].T), np.median(sampler.chain[:, :, 1].T), np.median(sampler.chain[:, :, 2].T), \
                              np.median(sampler.chain[:, :, 3].T), np.median(sampler.chain[:, :, 4].T), np.median(sampler.chain[:, :, 5].T), \
                              np.median(sampler.chain[:, :, 6].T), np.median(sampler.chain[:, :, 7].T), np.median(sampler.chain[:, :, 8].T), \
                              np.median(sampler.chain[:, :, 9].T), np.median(sampler.chain[:, :, 10].T), np.median(sampler.chain[:, :, 11].T)], rasterized=True)#, \
                              #np.median(sampler.chain[:, :, 12].T)])#, np.median(sampler.chain[:, :, 13].T)])
fig.show()
