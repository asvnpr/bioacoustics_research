import numpy
from pylab import *

# index 0 ->  1 for 86.021505376344081hz   0.0058125 segs
# index 1 ->  2 for 43.01075268817204hz    0.011625 segs
# index 2 ->  4 for 21.50537634408602hz    0.02325 segs
# index 3 ->  8 for 10.75268817204301hz    0.0465 segs
# index 4 ->  16 for 5.376344086021505hz     0.09299... segs
multiplier = [1,2,4,8,16]
bdws = [86.021505376344081,43.01075268817204,21.50537634408602,10.75268817204301,5.376344086021505]
binsRequired = 16.0

def get_nfft(sRate,mindex=0):
    nfft = 1116*multiplier[mindex]
    if float(sRate) == 16000.0:
        nfft = 93*multiplier[mindex]
    if float(sRate) == 32000.0:
        nfft = 186*multiplier[mindex]
    if float(sRate) == 48000.0:
        nfft = 279*multiplier[mindex]
    if float(sRate) == 96000.0:
        nfft = 558*multiplier[mindex]
    return nfft

def get_freqs(mindex=0):
    nft=1116*multiplier[mindex]
    x = numpy.random.rand(192000)
    Pxx, freqs, bins = mlab.specgram(x,NFFT=nft*2,Fs=192000,noverlap=nft)
    return freqs[1:]

def band2index(band):
    binBan = float(band)/binsRequired
    i = 0
    for b in bdws:
        if b < binBan:
            return i
        i = i + 1
    
    return i

