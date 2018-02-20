"""
Utility functions for scatter fitting routines
"""

import numpy as np

def writePDVtimeSeries(dspec, freqs, tInt, nch=1, src='FRB', ofn='timeseries.ascii'): 
    """A function to mimic a simple time series output of running pdV -kAT,
    assumes nsub=1, npol=1, nbin based on input dynamic spectrum
    This is used to input into Marisa's code

    dspec: 2-d numpy array, dynamic spectrum (time, freq)
    freqs: 1-d numpy array, frequencies in MHz
    tInt: float, integration time in seconds
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
        hdrStr += '\n# MJD(mid): 56384.680540767529 Tsub: %f Freq: %f BW: %f'%(tInt*1e6, decfreqs[chIdx], bw) # TODO: is the Tsub in microseconds?
        
        #write ascii profile
        # nsub nch nbin stokesI
        # 0 1 436 62.855
        for tIdx in np.arange(decdspec.shape[0]):
            hdrStr += '\n0 %i %i %f'%(chIdx, tIdx, decdspec[tIdx, chIdx])

    ofh = open(ofn, 'w')
    ofh.write(hdrStr)
    ofh.close()

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

