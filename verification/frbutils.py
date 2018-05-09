"""
Utility functions to FRB verification criteria
"""

import numpy as np
import filterbankio
import dedispersion

def deltat(dm, fStart, fStop):
    """returns delta t in ms
    fStart: MHz
    fStop: MHz"""
    return dm * (4.15 * (10.**6) * (fStart**(-2.) - fStop**(-2.)))

def gaussianFilter(arrShape, tSigma, fSigma):
    # Gaussian filter
    # arrShape: 2-D array shape
    # tSigma: sigma in time (ms)
    # fSigma: sigma in freq (MHz)
    lpos, mpos = np.mgrid[0:arrShape[0],0:arrShape[1]]
    taper = np.exp(-1. * ( (((lpos - (arrShape[0]/2.))**2.) / (2. * tSigma**2.)) + \
                           (((mpos - (arrShape[1]/2.))**2.) / (2. * fSigma**2.)) ))
    return taper

def convolveTaper(gaussImg, img):
    return np.fft.fftshift(np.fft.ifft2(np.fft.fft2(gaussImg) * np.fft.fft2(img))).real

def dynamicSpectrum(filfn, start_time, time_window, timeFactor=1, freqFactor=1, dm=0., applyGauss=False, fGauss=1., tGauss=1., rfi=[]):
    """Return a time-frequency cut-out from a filterbank file
    filfn: str, filterbank filename
    start_time: float, start time in seconds
    time_window: float, time window in seconds
    timeFactor: int, decimation factor
    freqFactor: int, decimation factor
    dm: float, dispersion measure
    applyGauss: bool, apply a Gaussian filter
    fGauss: float, Gaussian filte in frequency
    tGauss: float, Gaussian filte in time
    rfi: list of int pairs, start and stop index of freq channels to replace with noise

    returns: 2-D float array
    """
    fil = filterbankio.Filterbank(filfn)

 # take care of signed 8-bit data (if needed)  # Kaustubh Rajwade
    for i in range(len(fil.data)):
      for j in range(40):
       if (fil.data[i,0][j] < 0.0):
        fil.data[i,0][j] = fil.data[i,0][j] + 255

    tInt = fil.header['tsamp'] # get tInt
    freqsHz = fil.freqs * 1e6 # generate array of freqs in Hz

    waterfall = np.reshape(fil.data, (fil.data.shape[0], fil.data.shape[2])) # reshape to (n integrations, n freqs)

    ddwaterfall = dedispersion.incoherent(freqsHz, waterfall, tInt, dm, boundary='wrap') # apply dedispersion

	# Time Decimation
	# average down by N time samples
    if waterfall.shape[0] % timeFactor==0:
	    decwaterfall = waterfall.reshape(waterfall.shape[0]/timeFactor, timeFactor, waterfall.shape[1]).mean(axis=1)
	    decddwaterfall = ddwaterfall.reshape(ddwaterfall.shape[0]/timeFactor, timeFactor, ddwaterfall.shape[1]).mean(axis=1)
	    tInt *= timeFactor
    else:
	    print 'WARNING: %i time samples is NOT divisible by %i, zero-padding spectrum to usable size'%(waterfall.shape[0], timeFactor)
	    zeros = np.zeros((timeFactor - (waterfall.shape[0] % timeFactor), waterfall.shape[1]))
	    decwaterfall = np.concatenate((waterfall, zeros))
	    decddwaterfall = np.concatenate((ddwaterfall, zeros))
	    decwaterfall = decwaterfall.reshape(decwaterfall.shape[0]/timeFactor, timeFactor, decwaterfall.shape[1]).mean(axis=1)
	    decddwaterfall = decddwaterfall.reshape(decddwaterfall.shape[0]/timeFactor, timeFactor, decddwaterfall.shape[1]).mean(axis=1)
	    tInt *= timeFactor
	
	# Frequency Decimation
    if decwaterfall.shape[1] % freqFactor==0:
	    decwaterfall = decwaterfall.reshape(decwaterfall.shape[0], decwaterfall.shape[1]/freqFactor, freqFactor).mean(axis=2)
	    decddwaterfall = decddwaterfall.reshape(decddwaterfall.shape[0], decddwaterfall.shape[1]/freqFactor, freqFactor).mean(axis=2)
	    freqsHz = freqsHz[::freqFactor]
    else:
	    print 'WARNING: %i frequency channels is NOT divisible by %i, ignoring option'%(decwaterfall.shape[1], freqFactor)

    # cut out region
    if start_time is None: startIdx = 0
    else: startIdx = int(start_time / tInt)
	
    if time_window is None:
	    endIdx = decwaterfall.shape[0]
    else:
	    endIdx = startIdx + int(time_window / tInt)
	    if endIdx > decwaterfall.shape[0]:
	        print 'Warning: time window (-w) in conjunction with start time (-s) results in a window extending beyond the filterbank file, clipping to maximum size'
	        endIdx = decwaterfall.shape[0]
	
    # RFI replacement
    rfiMask = np.zeros_like(decddwaterfall)
    for freqPair in rfi:
        rfiMask[:,freqPair[0]:freqPair[1]] = 1.
    dsMean = np.ma.array(decddwaterfall, mask=rfiMask).mean()
    dsStd = np.ma.array(decddwaterfall, mask=rfiMask).std()
    for freqPair in rfi:
        decddwaterfall[:,freqPair[0]:freqPair[1]] = np.random.normal(dsMean, dsStd, size=decddwaterfall[:,freqPair[0]:freqPair[1]].shape)
        decwaterfall[:,freqPair[0]:freqPair[1]] = np.random.normal(dsMean, dsStd, size=decwaterfall[:,freqPair[0]:freqPair[1]].shape)
        #decddwaterfall[:,freqPair[0]:freqPair[1]] = np.zeros(decddwaterfall[:,freqPair[0]:freqPair[1]].shape)
        #decwaterfall[:,freqPair[0]:freqPair[1]] = np.zeros(decwaterfall[:,freqPair[0]:freqPair[1]].shape)

    decwaterfall = decwaterfall[startIdx:endIdx,:]
    decddwaterfall = decddwaterfall[startIdx:endIdx,:]

    if applyGauss:
        gaussFilter = gaussianFilter(decddwaterfall.shape, tGauss, fGauss)
        decddwaterfall = convolveTaper(gaussFilter, decddwaterfall)
        decwaterfall = convolveTaper(gaussFilter, decwaterfall)

    return decwaterfall, decddwaterfall, tInt, freqsHz / 1e6

def flux(SEFD, snr, nAnt, nPol, tObs, nuObs):
    """
    SEFD: float, system-equivalent flux density
    snr: float, SNR of pulse
    nAnt: int, number of antennas/elements in the telescope array
    nPol, int, number of polarizations in the reciever (1 or 2)
    tObs: float, observational integration time, in seconds
    nuObs: float, observational bandwidth, in Hz
    """
    return SEFD * snr / (nAnt * np.sqrt(nPol * tObs * nuObs))

def dmSpace(filfn, start_time, time_window, minDM, maxDM, dmStep, timeFactor=1, rfi=[]):
    """Return a time-frequency cut-out from a filterbank file
    filfn: str, filterbank filename
    start_time: float, start time in seconds
    time_window: float, time window in seconds
    timeFactor: int, decimation factor
    minDM: int, minimum DM trial
    maxDM: int, maximum DM trial
    dmStep: int, trail step size
    rfi: list of int pairs, start and stop index of freq channels to replace with noise

    returns: 2-D float array
    """
    fil = filterbankio.Filterbank(filfn)

# Signed 8-bit data
    for i in range(len(fil.data)):
          for j in range(40):
                 if (fil.data[i,0][j] < 0.0):
                         fil.data[i,0][j] = fil.data[i,0][j] + 255
    tInt = fil.header['tsamp'] # get tInt
    freqsHz = fil.freqs * 1e6 # generate array of freqs in Hz

    print 'Maximum delay (ms) based on maximum DM (%f):'%maxDM, deltat(maxDM, freqsHz[0]/1e6, freqsHz[-1]/1e6)
    
    testDMs = np.arange(minDM, maxDM, dmStep)

    waterfall = np.reshape(fil.data, (fil.data.shape[0], fil.data.shape[2])) # reshape to (n integrations, n freqs)

    # cut out region
    if start_time is None: startIdx = 0
    else: startIdx = int(start_time / tInt)
	
    if time_window is None:
	    endIdx = waterfall.shape[0]
    else:
	    endIdx = startIdx + int(time_window / tInt)
	    if endIdx > waterfall.shape[0]:
	        print 'Warning: time window (-w) in conjunction with start time (-s) results in a window extending beyond the filterbank file, clipping to maximum size'
	        endIdx = waterfall.shape[0]
	
    # RFI replacement
    rfiMask = np.zeros_like(waterfall)
    for freqPair in rfi:
        rfiMask[:,freqPair[0]:freqPair[1]] = 1.
    dsMean = np.ma.array(waterfall, mask=rfiMask).mean()
    dsStd = np.ma.array(waterfall, mask=rfiMask).std()
    for freqPair in rfi:
        waterfall[:,freqPair[0]:freqPair[1]] = np.random.normal(dsMean, dsStd, size=waterfall[:,freqPair[0]:freqPair[1]].shape)

    waterfall = waterfall[startIdx:endIdx,:]

    dmSpaceArr = np.zeros((testDMs.shape[0], waterfall.shape[0]))

    for dmid, dm in enumerate(testDMs):
        dmSpaceArr[dmid, :] = np.mean(dedispersion.incoherent(freqsHz, waterfall, tInt, dm, boundary='wrap'), axis=1)

	# Time Decimation
	# average down by N time samples
    if dmSpaceArr.shape[1] % timeFactor==0:
	    decdmSpaceArr = dmSpaceArr.reshape(dmSpaceArr.shape[0], dmSpaceArr.shape[1]/timeFactor, timeFactor).mean(axis=2)
	    tInt *= timeFactor
    else:
        print 'WARNING: %i time samples is NOT divisible by %i, zero-padding spectrum to usable size'%(dmSpaceArr.shape[1], timeFactor)
        zeros = np.zeros((dmSpaceArr.shape[0], timeFactor - (dmSpaceArr.shape[1] % timeFactor)))
        decdmSpaceArr = np.concatenate((dmSpaceArr, zeros), axis=1)
        decdmSpaceArr = decdmSpaceArr.reshape(decdmSpaceArr.shape[0], decdmSpaceArr.shape[1]/timeFactor, timeFactor).mean(axis=2)
        decdmSpaceArr = decdmSpaceArr[:,:-1] # drop last integration
        tInt *= timeFactor

    return decdmSpaceArr, tInt, freqsHz / 1e6

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    #frbFil = '/local/griffin/data/FRB/FRB130729/FRB130729_s41r10b8.fil'
    ## dynamic spectrum parameters
    #timeFactor = 1 # time decimation factor
    #freqFactor = 1 # frequency decimation factor
    #dm = 851.5 # dedispersion
    #
    #start_time = 5.1 # seconds
    #time_window = 0.2 # seconds
    #
    #applyGauss = True # apply a Gaussian filter to the dynamic specturm
    #fGauss = 8. # frequency bins
    #tGauss = 8. # time bins
    #
    #decwaterfall, decddwaterfall, tInt, freqs = dynamicSpectrum(frbFil, start_time, time_window,
    #                                    timeFactor=timeFactor, freqFactor=freqFactor,
    #                                    dm=dm,
    #                                    applyGauss=applyGauss, fGauss=fGauss, tGauss=tGauss,
    #                                    rfi=[[200,250]])
    
    #ddTimeSeries = np.sum(decddwaterfall, axis=1)
    #
    #cmap = 'magma'
    #
    #fig = plt.figure(figsize=(12,6)) # (width, height)
    #
    #ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3, rowspan=2)
    #
    ## crop out RFI flagged high freqs
    #imRaw = plt.imshow(np.flipud(decddwaterfall[:,:850].T), aspect='auto', cmap=plt.get_cmap(cmap), interpolation='nearest')
    #
    #plt.ylabel('MHz')
    #ax1.get_xaxis().set_visible(False)
    #
    #ax2 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
    #lineColor = 'k'
    #plt.plot(1000.*np.arange(decddwaterfall.shape[0]), ddTimeSeries, lineColor, alpha=0.8)
    #plt.xlim(0, 1000.*ddTimeSeries.shape[0])
    #plt.title('DM=%0.f'%dm)
    #plt.xlabel('ms')
    #ax2.get_yaxis().set_visible(False)
    #
    #plt.tight_layout()
    #plt.show()
   
    ###############################
    #frbFil = '/local/griffin/data/alfaburst/priorityModel/B1859+03/Beam0_fb_D20150801T220510.buffer11.fil'
    #timeFactor = 4 # time decimation factor
    #
    #start_time = 3.9
    #time_window = 0.4

    #minDM = 0
    #maxDM = 1000
    #dmStep = 4

    ###############################
    #frbFil = '/local/griffin/data/FRB/FRB130729/FRB130729_s41r10b8.fil'
    #timeFactor = 8 # time decimation factor
    #
    #start_time = 5.0 # seconds
    #time_window = 1.5 # seconds

    #minDM = 840
    #maxDM = 870
    #dmStep = 0.2
    #
    #dmSpaceArr, tInt, freqs = dmSpace(frbFil, start_time, time_window, minDM, maxDM, dmStep, timeFactor, rfi=[[200,250], [850,1023]])
    #print dmSpaceArr.shape
    #
    #cmap = 'viridis'
    #fig = plt.figure(figsize=(12,6)) # (width, height)
    #
    #plt.imshow(np.flipud(dmSpaceArr), aspect='auto', extent=(0, tInt*dmSpaceArr.shape[1], minDM, maxDM), cmap=plt.get_cmap(cmap), interpolation='nearest')
    ##for idx in np.arange(dmSpaceArr.shape[0]):
    ##    plt.plot(dmSpaceArr[idx])
    #plt.ylabel('DM')
    #plt.xlabel('t (s)')
    #plt.colorbar(fraction=0.025)
    #plt.show()

