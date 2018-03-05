"""
Utility functions for scatter fitting routines
"""

import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model, conf_interval, printfuncs
from lmfit.models import GaussianModel, PowerLawModel 
from lmfit import Model, conf_interval, printfuncs
from lmfit import minimize, Parameter, Parameters, fit_report
#from lmfit.models import LinearModel, PowerLawModel, ExponentialModel, QuadraticModel 

def writePDVtimeSeries(dspec, freqs, tspan, nch=1, src='FRB', ofn='timeseries.ascii'): 
    """A function to mimic a simple time series output of running pdV -kAT,
    assumes nsub=1, npol=1, nbin based on input dynamic spectrum
    This is used to input into Marisa's code

    dspec: 2-d numpy array, dynamic spectrum (time, freq)
    freqs: 1-d numpy array, frequencies in MHz
    tspan: float, obs time in seconds
    nch: int, number of channels to output
    src: string, source name
    ofn: string, output file name
    """
    fDecFactor = int(freqs.shape[0] / nch) # frequency decimation factor
    decdspec = np.zeros((dspec.shape[0], nch))
    if dspec.shape[1] % fDecFactor > 0:
        print 'WARNING: the number of frequency channels %i is not an integer multiple of the number of subband channels %i, %i frequency channels will be dropped'%(dspec.shape[1], nch, dspec.shape[1] % fDecFactor)
        dspecClip = dspec[:,:dspec.shape[1] - (dspec.shape[1] % fDecFactor)]
        freqsClip = freqs[:dspec.shape[1] - (dspec.shape[1] % fDecFactor)]
    else:
        dspecClip = dspec.copy()
        freqsClip = freqs.copy()
    dspecClip -= dspecClip.mean()

    # average down in frequency
    decdspec = np.mean(dspecClip.reshape(dspecClip.shape[0], nch, fDecFactor), axis=2)
    decfreqs = np.mean(freqsClip.reshape(nch, fDecFactor), axis=1)
    bw = freqs[fDecFactor] - freqs[0]

    # overall header
    # # File: B0611+22_L116889 Src: J0614+2229 Nsub: 1 Nch: 8 Npol: 1 Nbin: 1024 RMS: 93.8508
    hdrStr = '# File: %s Src: %s Nsub: 1 Nch: %i Npol: 1 Nbin: %i RMS: %f'%(src, src, nch, decdspec.shape[0], np.std(decdspec))

    for chIdx in np.arange(nch):
        # write channel header
        # # MJD(mid): 56384.680540767529 Tsub: 599.953 Freq: 114.752 BW: 9.76562
        # NOTE: MJD is fixed, and not the real MJD
        hdrStr += '\n# MJD(mid): 56384.680540767529 Tsub: %f Freq: %f BW: %f'%(tspan, decfreqs[chIdx], bw)
        
        #write ascii profile
        # nsub nch nbin stokesI
        # 0 1 436 62.855
        for tIdx in np.arange(decdspec.shape[0]):
            hdrStr += '\n0 %i %i %f'%(chIdx, tIdx, decdspec[tIdx, chIdx])

    ofh = open(ofn, 'w')
    ofh.write(hdrStr)
    ofh.close()
    

    
### Read ascii files

def read_headerfull(filepath):
    f = open(filepath)
    lines = f.readlines()
    header0 = lines[0]
    header1 = lines[1]
    h0_lines = header0.split()
    if h0_lines[0] == '#':
        h0_lines = h0_lines[1:len(h0_lines)]
    else:
        h0_lines = h0_lines    
    file_name = h0_lines[1]
    pulsar_name = h0_lines[3]
    nsub = int(h0_lines[5])
    nch = int(h0_lines[7])
    npol = int(h0_lines[9])
    nbins = int(h0_lines[11])
    rms = float(h0_lines[13])
    h1_lines = header1.split()
    tsub = float(h1_lines[4])  
#    return file_name, pulsar_name, nsub, nch, npol, nbins, rms
    return pulsar_name, nch, nbins, nsub, rms, tsub



def read_data(filepath, profilenumber, nbins):
    d = open(filepath)
    lines = d.readlines()
    
    profile_start = 2+profilenumber*(nbins+1)
    profile_end = profile_start + nbins
    
    lines_block = lines[profile_start:profile_end]

    if lines[profile_start-1].split()[0] == '#':
        freqc = float(lines[profile_start-1].split()[6])
        bw = float(lines[profile_start-1].split()[8])
        freqm = 10**((np.log10(freqc+ bw/2.)+ np.log10(freqc - bw/2.))/2)
    else:
        freqc = float(lines[profile_start-1].split()[5])
        bw = float(lines[profile_start-1].split()[7])
        freqm = 10**((np.log10(freqc+ bw/2.)+ np.log10(freqc - bw/2.))/2)
    datalist = []
    for i in range(nbins):
        data= float(lines_block[i].split()[3])
        datalist.append(data)

    return np.array(datalist), freqc, freqm

    
    
    

def find_rms(data,nbins):
    windowsize = 32 
    windows = int(nbins/windowsize)
    rms_loc = np.zeros(windows)
    for i in range(windows):
        start = i*windowsize
        end = start + windowsize
        rms_loc[i] = np.std(data[start:end])
    return np.min(rms_loc)

def smooth(y, box_pts):
    gauss = np.ones(box_pts)
#    box = np.ones(box_pts)/box_pts
    sigma = (1./6.)*box_pts
    mean = box_pts/2.
    for i in range(box_pts):
        gauss[i] = (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-(i-mean)**2/(2*sigma**2))
    y_smooth = np.convolve(y, gauss, mode='same')
    return y_smooth


def find_peaksnr_smooth(data,rms):
    boxsize = int(0.05*len(data))
    smootheddata = smooth(data,boxsize)
    peak = np.max(smootheddata)
    snr = peak/rms
    return snr

def find_peaksnr(data,rms):
    peak = np.max(data)
    snr = peak/rms
    return snr 

def tauatfreq(oldfreq,oldtau,newfreq,specindex):
    newtau = oldtau*(newfreq)**(-specindex)/(oldfreq**(-specindex))
    return newtau

def makeprofile(nbins = 2**9, ncomps = 1, amps = 1, means = 100, sigmas = 10):
    if ncomps == 1:
        npamps = np.array([amps])
        npmeans = np.array([means])
        npsigmas = np.array([sigmas])
    else:
        npamps = np.array(amps)
        npmeans = np.array(means)
        npsigmas = np.array(sigmas)

    profile = np.zeros(nbins)
    x = np.linspace(1,nbins,nbins)

    for i in range(ncomps):
#        print npmeans[i]
        profile = profile + \
        npamps[i]*np.exp(-pow(x-npmeans[i],2)/(2*pow(npsigmas[i],2)))
    return x, profile

def pulsetrain(npulses = 10, bins = np.linspace(1,512,512), profile = np.zeros(512)):
    nbins = np.max(bins)
    train = np.zeros(npulses*int(nbins))
    nbins = int(nbins)
    for i in range(npulses):
        startbin = int(i*nbins)
        endbin = int(startbin + nbins)
        train[startbin:endbin] = profile
    return train

def pulsetrain_bins(npulses, numberofbins, profile):
    binsrange = np.linspace(1,numberofbins,numberofbins)
    nbins = np.max(binsrange)
    nbins = int(nbins)
#    print nbins
    train = np.zeros(npulses*int(nbins))

    for i in range(npulses):
        startbin = int(i*nbins)
        endbin = int(startbin + nbins)
        train[startbin:endbin] = profile
    return train

def psrscatter(brfunc, profile):
    scattered = np.convolve(profile,brfunc)
    profint = np.sum(profile)
    scint = np.sum(scattered)
    scatterednorm = scattered / scint * profint
    bins = profile.shape[0]
    out = scatterednorm[0:bins]
    return out

def psrscatter_noconserve(brfunc, profile):
    scattered = np.convolve(profile,brfunc)
    bins = profile.shape[0]
    out = scattered[0:bins]
    return out
    
def step(x):
    return 1 * (x >= 0)

### Broadening functions
# 1. Isotropic scattering

def broadfunc(x,tau):
    tau = float(tau)
    broadfunc = (1/tau)*np.exp(-x/tau)*step(x)
    return broadfunc   

# 2. Extremely anisotropic scattering

def broadfunc1D(x,tau):
    broadfunc1 = (1.0/np.sqrt(x*tau*pi))*np.exp(-x/tau)
    return broadfunc1


def extractpulse(train, pulsesfromend, binsperpulse):
    if pulsesfromend == 0:
        start = 0
        end = binsperpulse
        zerobpulse = train[start:end]-np.min(train[start:end])
        rectangle = np.min(train[start:end])*binsperpulse
        flux = np.sum(train[start:end]) - rectangle
        return train[start:end], zerobpulse, rectangle, flux

    else:
        start = -pulsesfromend*binsperpulse
        end = start + binsperpulse
        zerobpulse = train[start:end]-np.min(train[start:end])
        rectangle = np.min(train[start:end])*binsperpulse
        flux = np.sum(train[start:end]) - rectangle
        return train[start:end], zerobpulse, rectangle, flux

def peaksnr(x, profile, snr):
    bins = profile.shape[0]
    noise = np.random.random(bins)
    peak = np.max(profile)
    out = profile/peak * snr + noise
    return out


def GxETrain(x,mu,sigma, A, tau, dc, nbins):
#This model convolves a pulsetrain with a broadening function
#It extracts one of the last convolved profiles, subtracts the climbed baseline and then adds noise to it
    mu, sigma, A, tau = float(mu),float(sigma), float(A), float(tau)
    bins, profile = makeprofile(nbins = nbins, ncomps = 1, amps = A, means = mu, sigmas = sigma)
    binstau = np.linspace(1,nbins,nbins)
    scat = psrscatter_noconserve(broadfunc(binstau,tau),pulsetrain_bins(3, nbins, profile))
#    scat = psrscatter(broadfunc(binstau,tau),pulsetrain_bins(3, nbins, profile))  
    climb, observed_nonoise, rec, flux = extractpulse(scat, 2, nbins)
    return observed_nonoise + dc
    
def GxETrain1D(x,mu, sigma, A, tau1, dc, nbins):
    mu, sigma, A, tau1 = float(mu),float(sigma), float(A), float(tau1)
    bins, profile = makeprofile(nbins = nbins, ncomps = 1, amps = A, means = mu, sigmas = sigma)
    binstau = np.linspace(1,nbins,nbins)
    scat = psrscatter(broadfunc1D(binstau,tau1),pulsetrain(3, bins, profile))
    climb, observed_nonoise, rec,flux = extractpulse(scat, 2, nbins)
    return observed_nonoise + dc
    
    
    
def tau_fitter(data,nbins):
    profile_peak = np.max(data)
    binpeak = np.argmax(data)  
    modelname = GxETrain
    model = Model(modelname)
                 
    model.set_param_hint('nbins', value=nbins, vary=False)            
    model.set_param_hint('sigma', value=15, vary=True, min =0, max = nbins)
    model.set_param_hint('mu', value=binpeak, vary=True, min=0, max = nbins)
    model.set_param_hint('A',value=profile_peak, vary=True, min=0)
    model.set_param_hint('tau',value=200, vary=True, min=0)
    model.set_param_hint('dc',value = 0, vary = True)
    pars = model.make_params()
    xax=np.linspace(1,nbins,nbins)

    #"""Fit data"""
    result = model.fit(data,pars,x=xax)
    print(result.fit_report(show_correl = True))
    
    noiselessmodel = result.best_fit
    besttau = result.best_values['tau']
    taustd = result.params['tau'].stderr  ##estimated 1 sigma error

    bestsig = result.best_values['sigma']
    bestmu = result.best_values['mu']
    bestA = result.best_values['A']
    bestdc = result.best_values['dc']
    
    bestsig_std = result.params['sigma'].stderr
    bestmu_std = result.params['mu'].stderr
    bestA_std = result.params['A'].stderr
    bestdc_std = result.params['dc'].stderr    
    
    bestparams = np.array([bestsig,bestmu,bestA,bestdc])
    bestparams_std = np.array([bestsig_std,bestmu_std,bestA_std,bestdc_std])
    
    """correlations with sigma"""    
    corsig = result.params['sigma'].correl
    #corA = result.params['A'].correl
    #corlist = [corsig,corA]
    
    
    rchi = result.redchi
    #return best values and std errors on the other parameters as well    
    
    return result, noiselessmodel, besttau, taustd, bestparams, bestparams_std, rchi, corsig


def tau_1D_fitter(data,nbins):

    profile_peak = np.max(data)
    binpeak = np.argmax(data)
    modelname = GxETrain1D
    model = Model(modelname)

    model.set_param_hint('nbins', value=nbins, vary=False)
    model.set_param_hint('sigma', value=15, vary=True, min =0, max = nbins)
    model.set_param_hint('mu', value=binpeak, vary=True, min=0, max = nbins)
    model.set_param_hint('A',value=profile_peak, vary=True,min=0)
    model.set_param_hint('tau1',value=200, vary=True, min=0)
#    model.set_param_hint('tau1',value=166.792877, vary=False)
    model.set_param_hint('dc',value = 0, vary = True)
    pars = model.make_params()

    result = model.fit(data,pars,x=np.linspace(1,nbins,nbins))
#    print(result.fit_report(show_correl = False))

    noiselessmodel = result.best_fit
    besttau = result.best_values['tau1']
    taustd = result.params['tau1'].stderr  ##estimated 1 sigma error

    bestsig = result.best_values['sigma']
    bestmu = result.best_values['mu']
    bestA = result.best_values['A']
    bestdc = result.best_values['dc']

    bestsig_std = result.params['sigma'].stderr
    bestmu_std = result.params['mu'].stderr
    bestA_std = result.params['A'].stderr
    bestdc_std = result.params['dc'].stderr

    bestparams = np.array([bestsig,bestmu,bestA,bestdc])
    bestparams_std = np.array([bestsig_std,bestmu_std,bestA_std,bestdc_std])

    """correlations with sigma"""
    corsig = result.params['sigma'].correl
    #corA = result.params['A'].correl
    #corlist = [corsig,corA]

    rchi = result.redchi

    return result, noiselessmodel, besttau, taustd, bestparams, bestparams_std, rchi, corsig


    
       
    
    
    
def produce_taufits(filepath,meth='iso'):
        pulsar, nch, nbins,nsub, lm_rms, tsub = read_headerfull(filepath)

        print0 = "Pulsar name: %s" %pulsar
        print1 = "Number of channels: %d" %nch
        print2 = "Number of bins: %d" %nbins
        print3 = "RMS: %f" %lm_rms
        print4 = "Tsub: %f sec" %tsub 
        for k in range(4):
              print eval('print{0}'.format(k))


        ## Define time axis, and time/bins conversions

        print "Using Tsub in header to convert bins to time. Note Tsub here is full phase time, corresponding to nbins."
        pulseperiod = tsub

        profilexaxis = np.linspace(0,pulseperiod,nbins)
        pbs = pulseperiod/nbins
        tbs = tsub/nbins

        obtainedtaus = []
        lmfittausstds = []

        freqmsMHz =[]
        freqcsMHz =[]
        noiselessmodels =[]
        results = []
        datas = []
        
        halfway = nbins/2.

        for i in range(nch):
            print "\n Channel %d" %i
            # read in data 
            data, freqc, freqm = read_data(filepath,i,nbins)
            freqmsMHz.append(freqm)
            freqcsMHz.append(freqc)
            # roll the data of lowest freq channel to middle of bins 
            if i ==0:
                peakbin = np.argmax(data)
                shift = int(halfway -int(peakbin))
                print 'peak bin at lowest freq channel:%d' %peakbin
            else:
                peakbin = peakbin
                shift = int(halfway - int(peakbin))
            data = np.roll(data,shift)
            print "Rolling data by -%d bins" %shift
            comp_rms = find_rms(data,nbins)

            if meth is None:
                        print "No fitting method was chosen. Will default to an isotropic fitting model. \n Use option -m with 'onedim' to change."
                        result, noiselessmodel, besttau, taustd, bestparams, bestparams_std, redchi, corsig = tau_fitter(data,nbins)

            elif meth == 'iso':
                        result, noiselessmodel, besttau, taustd, bestparams, bestparams_std, redchi, corsig = tau_fitter(data,nbins)

            elif meth == 'onedim':
                        result, noiselessmodel, besttau, taustd, bestparams, bestparams_std, redchi, corsig = tau_1D_fitter(data,nbins)         

            comp_SNR_model = find_peaksnr(noiselessmodel,comp_rms)

            print 'Estimated SNR (from model peak and rms): %.2f' % comp_SNR_model
            comp_SNR =  find_peaksnr_smooth(data,comp_rms)
            print 'Estimated SNR (from data peak and rms): %.2f' % comp_SNR

            print 'Channel Tau (ms): %.2f \pm %.2f ms' %(besttau,taustd)

            obtainedtaus.append(besttau)
            lmfittausstds.append(taustd)
            noiselessmodels.append(noiselessmodel)
            results.append(result)

            datas.append(data)

        ## insert a per channel SNR cutoff here if want to.
        ## have removed it for now

        print "Using no SNR cutoff"

        data_highsnr =np.array(datas)
        model_highsnr = np.array(noiselessmodels)

        taus_highsnr = np.array(obtainedtaus)
        lmfitstds_highsnr = np.array(lmfittausstds)

        taussec_highsnr = taus_highsnr*pbs
        lmfitstdssec_highsnr = lmfitstds_highsnr*pbs

        freqMHz_highsnr = np.array(freqmsMHz)
        number_of_plotted_channels = len(data_highsnr)
        npch = number_of_plotted_channels


        """Plotting starts"""

        #plot onedim in blue dashed
        #else plot in red
        if meth == 'onedim':
            prof = 'b--'
            lcol='b'
        else:
            prof = 'r-'
            lcol ='r'

        ##PLOT PROFILES##  

        numplots = int(np.ceil(npch/4.))

        """Compute residuals"""

        resdata = data_highsnr - model_highsnr
        resnormed = (resdata-resdata.mean())/resdata.std()


       #"""Plot 1: Pulse profiles and fits"""
        if taussec_highsnr[0] > 1:
            taulabel =  taussec_highsnr
            taulabelerr = lmfitstdssec_highsnr
            taustring = 'sec'
        else:
            taulabel = taussec_highsnr*1000
            taulabelerr = lmfitstdssec_highsnr*1000
            taustring = 'ms'

        for k in range(numplots):
            j = 4*k
            figg = plt.figure(k+1,figsize=(10,8))
            for i in range(np.min([4,npch])):
                figg.subplots_adjust(left = 0.08, right = 0.98, wspace=0.35,hspace=0.35,bottom=0.15)
                #plt.rc('text', usetex=True)
                plt.rc('font', family='serif')              
                plt.subplot(2,2,i+1)
                plt.plot(profilexaxis,data_highsnr[j+i],'k',alpha = 0.30)
                plt.plot(profilexaxis,model_highsnr[j+i],prof,lw = 2.0, alpha
                        = 0.7,label=r'$\tau: %.2f \pm %.2f$ %s'
                        %(taulabel[j+i], taulabelerr[j+i], taustring))
                plt.title('%s at %.1f MHz' %(pulsar, freqMHz_highsnr[j+i]))
                plt.ylim(ymax=1.3*np.max(data_highsnr[j+i]))
                plt.xlim(xmax=pulseperiod)
                plt.xticks(fontsize=11)
                plt.yticks(fontsize=11)
                plt.xlabel('time (s)',fontsize=11)
                plt.legend(fontsize=11,numpoints=1)
                plt.ylabel('normalized intensity',fontsize=11)
        plt.show()

        for i in range(nch):
            print'Tau (ms): %.2f' %(1000*taussec_highsnr[i])
            tau1GHz = tauatfreq(freqMHz_highsnr[i]/1000.,taussec_highsnr[i],1.0,4)
            print 'tau1GHz_alpha_4 (ms) ~ %.2f \n' %(tau1GHz*1000)
            
        return freqMHz_highsnr, taussec_highsnr, lmfitstdssec_highsnr 


def produce_tauspectrum(freqMHz,taus,tauserr):
    freqGHz = freqMHz/1000.
    powmod = PowerLawModel()
    powparstau = powmod.guess(taus,x=freqGHz)
    
    #remove tauserr = 0 entries
    
    tauserr = tauserr[np.nonzero(tauserr)]
    taus = taus[np.nonzero(tauserr)]
    freqGHz = freqGHz[np.nonzero(tauserr)]
    freqMHz = freqMHz[np.nonzero(tauserr)]
    
    powout = powmod.fit(taus,powparstau,x=freqGHz,weights=1/(np.power(tauserr,2)))
    
    print(fit_report(powout.params))
    fit = powout.best_fit
    alpha = -powout.best_values['exponent']
    alphaerr = powout.params['exponent'].stderr

    fig = plt.figure()         
    plt.errorbar(freqMHz,taus,yerr=tauserr,fmt='k*',markersize=9.0,capthick=2,linewidth=1.5,label=r'$\alpha = %.1f \pm %.1f$' %(alpha,alphaerr))
    plt.plot(freqMHz,fit,'k--',linewidth=1.5)
    plt.xscale('log')
    plt.yscale('log')
    plt.yticks(fontsize=11)
    ticksMHz = (freqMHz).astype(np.int)[0:len(freqMHz):2]
    plt.xticks(ticksMHz,ticksMHz,fontsize=11)
    plt.legend(fontsize=11,numpoints=None)
    plt.xlabel(r'$\nu$ (MHz)',fontsize=11, labelpad=15.0)
    plt.ylabel(r'$\tau$ (sec)',fontsize=11)
    plt.xlim(xmin = 0.95*freqMHz[0],xmax=1.05*freqMHz[-1])
    plt.gcf().subplots_adjust(bottom=0.15)
    
    return alpha, alphaerr






if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import frbutils
    import seaborn as sns

    frbFil = '/local/griffin/data/FRB/FRB130729/FRB130729_s41r10b8.fil' # Filterbank file of FRB event (aslxlap07)
    
    ### dynamic spectrum parameters
    timeFactor = 1 # time decimation factor
    freqFactor = 1 # frequency decimation factor
    dm = 851.5 # dedispersion
    
    start_time = 5.1 # seconds
    time_window = 0.2 # seconds
    
    applyGauss = True # apply a Gaussian filter to the dynamic specturm
    fGauss = 4. # frequency bins
    tGauss = 8. # time bins
    
    rfi = [[45,65], [155,175], [200,250], [850,1023]] # frequency channels to flag as rfi, pairs are the edges of bands
    
    decwaterfall, decddwaterfall, tInt, freqs = frbutils.dynamicSpectrum(frbFil, start_time,
            time_window, timeFactor=timeFactor, freqFactor=freqFactor,
            dm=dm, applyGauss=applyGauss, fGauss=fGauss, tGauss=tGauss, rfi=rfi)

    #decddwaterfall = np.zeros((3125, 1024+1))
    #tInt = 6.4e-05
    #freqs = np.arange(1024)*(1182.5859375 - 1182.1953125) + 1182.1953125

    writePDVtimeSeries(decddwaterfall, freqs, tInt, nch=8, src='FRB130729', ofn='FRB130729.ascii')

    import DataReadIn
    print DataReadIn.read_header('FRB130729.ascii')

    #ddTimeSeries = np.mean(decddwaterfall, axis=1)
    #
    #cmap = 'magma'
    #sns.set_style('white', rc={'axes.linewidth': 0.5})
    #
    #fig = plt.figure(figsize=(12,6)) # (width, height)
    #
    #ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3, rowspan=2)
    #
    #imRaw = plt.imshow(np.flipud(decddwaterfall.T), extent=(0, tInt*decddwaterfall.shape[0],\
    #                        freqs[0], freqs[-1]), aspect='auto', cmap=plt.get_cmap(cmap), interpolation='nearest')
    ## crop out RFI flagged high freqs
    ##imRaw = plt.imshow(np.flipud(decddwaterfall[:,:850].T), extent=(0, tInt*decddwaterfall.shape[0],\
    ##                        freqs[0], freqs[850]), aspect='auto', cmap=plt.get_cmap(cmap), interpolation='nearest')
    #for rfiPair in rfi:
    #    plt.axhspan(freqs[rfiPair[0]/freqFactor], freqs[rfiPair[1]/freqFactor], alpha=0.4, color='y')
    #plt.ylabel('MHz')
    #
    #ax1.get_xaxis().set_visible(False)
    #
    #ax2 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
    #lineColor = 'k'
    #plt.plot(1000.*tInt*np.arange(decddwaterfall.shape[0]), ddTimeSeries, lineColor, alpha=0.8)
    #plt.xlim(0, 1000.*tInt*ddTimeSeries.shape[0])
    #plt.title('DM=%0.f'%dm)
    #plt.xlabel('ms')
    #ax2.get_yaxis().set_visible(False)
    #
    #plt.show()

