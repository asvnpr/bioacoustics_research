from pylab import *
import numpy

def get_freqs():
    nft=4464
    x = numpy.random.rand(192000)
    Pxx, freqs, bins = mlab.specgram(x,NFFT=nft*2,Fs=192000,noverlap=nft)
    return freqs[1:]
