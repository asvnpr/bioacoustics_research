from rec import Rec
from a2audio.thresholder import Thresholder
from pylab import *
from matplotlib import *
import numpy
import math
import json
import a2pyutils.storage
from samplerates import *
analysis_sample_rates = [16000.0,32000.0,48000.0,96000.0,192000.0]

class Roizer:

    def __init__(self, uri ,tempFolder,storage ,iniSecs=5,endiSecs=15,lowFreq = 1000, highFreq = 2000,logs=None,useSsim=True,bIndex=0,save_model=True):
        
        if type(uri) is not str and type(uri) is not unicode:
            raise ValueError("uri must be a string")
        if type(tempFolder) is not str:
            raise ValueError("invalid tempFolder")
        if not os.path.exists(tempFolder):
            raise ValueError("invalid tempFolder")
        elif not os.access(tempFolder, os.W_OK):
            raise ValueError("invalid tempFolder")
        if not isinstance(storage, a2pyutils.storage.AbstractStorage):
            raise ValueError("invalid storage instance")
        if type(iniSecs) is not int and  type(iniSecs) is not float:
            raise ValueError("iniSecs must be a number")
        if type(endiSecs) is not int and  type(endiSecs) is not float:
            raise ValueError("endiSecs must be a number")
        if type(lowFreq) is not int and  type(lowFreq) is not float:
            raise ValueError("lowFreq must be a number")
        if type(highFreq) is not int and  type(highFreq) is not float:
            raise ValueError("highFreq must be a number")
        if iniSecs>=endiSecs:
            raise ValueError("iniSecs must be less than endiSecs")
        if lowFreq>=highFreq :
            raise ValueError("lowFreq must be less than highFreq")
        self.spec = None
        recording = Rec(uri,tempFolder,storage,logs, removeFile=True , test=False,resample=False)
        self.logs = logs
        self.ssim = useSsim
        self.bIndex = bIndex
        self.save_model = save_model
        if self.logs:
            logs.write("Roizer: "+str(uri))
        if  'HasAudioData' in recording.status:
            #if float(recording.sample_rate) not in analysis_sample_rates:
            #    self.status = "SampleRateNotSupported"
            #    if self.logs:
            #        logs.write("Roizer: "+str(recording.sample_rate)+" is not supported")
            #    return None              
            self.original = recording.original
            self.sample_rate = recording.sample_rate
            self.recording_sample_rate = recording.sample_rate
            self.channs = recording.channs
            self.samples = recording.samples
            self.status = 'HasAudioData'
            self.iniT = iniSecs
            self.endT = endiSecs
            self.lowF = lowFreq
            self.highF = highFreq 
            self.uri = uri
            if self.logs:
                self.logs.write("Roizer: has audio data")          
        else:
            if self.logs:
                self.logs.write("Roizer: has no audio data")   
            self.status = "NoAudio"
            return None
        dur = float(self.samples)/float(self.sample_rate)
        if dur <= endiSecs-0.001:
            raise ValueError("endiSecs greater than recording duration")
        
        if  'HasAudioData' in self.status:
            if self.logs:
                self.logs.write("Roizer: creating spectrogram")
            try:
                self.spectrogram()
            except:
                if self.logs:
                    self.logs.write("Roizer: cannot create spectrogram")
                self.status = "NoSpectrogram"
                return None
            if self.logs:
                self.logs.write("Roizer: spectrogram done")  

    def getAudioSamples(self):
        return self.original
    
    def getSpectrogram(self):
        if self.spec is not None:
             self.spectrogram()
        return self.spec
    
    def spectrogram(self):
        initSample = int(math.floor(float((self.iniT)) * float((self.sample_rate))))
        endSample = int(math.floor(float((self.endT)) * float((self.sample_rate))))
        if endSample >= len(self.original):
           endSample = len(self.original) - 1

        maxHertzInRec = float(self.sample_rate)/2.0
        nfft = 512
        targetrows = 512

        data = self.original[initSample:endSample]
        Pxx, freqs, bins = mlab.specgram(data, NFFT=nfft*2, Fs=self.sample_rate, noverlap=nfft)
        dims =  Pxx.shape
        i =0
        while freqs[i] < self.lowF:
            Pxx[i,:] = 0 
            i = i + 1
        #calculate decibeles in the passband
        while freqs[i] < self.highF:
            Pxx[i,:] =  10. * numpy.log10(Pxx[i,:].clip(min=0.0000000001))
            i = i + 1
        #put zeros in unwanted frequencies (filter)
        while i <  dims[0]:
            Pxx[i,:] = 0
            i = i + 1

        Z = numpy.flipud(Pxx[0:(Pxx.shape[0]-1),:])

        #z = numpy.zeros(shape=(targetrows,Pxx.shape[1]))
        #z[(targetrows-Pxx.shape[0]+1):(targetrows-1),:] = Z
        self.spec = Z
 

