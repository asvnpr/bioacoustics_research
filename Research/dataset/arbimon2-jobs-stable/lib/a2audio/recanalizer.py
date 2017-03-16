from a2audio.rec import Rec
from a2audio.thresholder import Thresholder
from pylab import *
import numpy
numpy.seterr(all='ignore')
numpy.seterr(divide='ignore', invalid='ignore')
import time
from skimage.measure import structural_similarity as ssim
from scipy.stats import pearsonr as prs
from scipy.stats import kendalltau as ktau
from  scipy.spatial.distance import cityblock as ct
from scipy.spatial.distance import cosine as csn
import math
from a2pyutils.logger import Logger
import os
import json
from scipy.stats import *
from  scipy.signal import *
import warnings
from samplerates import *
import cv2
from cv import *
import a2pyutils.storage
from contextlib import closing
import random

analysis_sample_rates = [16000.0,32000.0,48000.0,96000.0,192000.0]


class Recanalizer:
    
    def __init__(self, uri, speciesSurface, low, high, tempFolder, storage, logs=None,test=False,useSsim = True,step=16,oldModel =False,numsoffeats=41,ransakit=False,bIndex=0,db=None,rec_id=None,job_id=None,model_type_id=4):
        if type(uri) is not str and type(uri) is not unicode:
            raise ValueError("uri must be a string")
        if type(speciesSurface) is not numpy.ndarray:
            raise ValueError("speciesSurface must be a numpy.ndarray. Input was a "+str(type(speciesSurface)))
        if type(low) is not int and  type(low) is not float:
            raise ValueError("low must be a number")
        if type(high) is not int and  type(high) is not float:
            raise ValueError("high must be a number")
        if low>=high :
            raise ValueError("low must be less than high")
        if type(tempFolder) is not str:
            raise ValueError("invalid tempFolder must be a string")
        if not os.path.exists(tempFolder):
            raise ValueError("invalid tempFolder does not exists")
        elif not os.access(tempFolder, os.W_OK):
            raise ValueError("invalid tempFolder")
        if not isinstance(storage, a2pyutils.storage.AbstractStorage):
            raise ValueError("invalid storage instance")
        if logs is not None and not isinstance(logs,Logger):
            raise ValueError("logs must be a a2pyutils.Logger object")
        start_time = time.time()
        self.distances = []
        self.low = float(low)
        self.high = float(high)
        self.columns = speciesSurface.shape[1]
        self.speciesSurface = speciesSurface
        self.logs = logs   
        self.uri = uri
        self.storage = storage
        self.tempFolder = tempFolder
        self.rec = None
        self.status = 'InitNoData'
        self.ssim = useSsim
        self.step = step
        self.oldModel = False
        self.numsoffeats = numsoffeats
        self.algo = 'sift'
        self.useRansac = ransakit
        self.hasrec = False
        self.bIndex = bIndex
        self.db =db
        self.rec_id = rec_id
        self.job_id = job_id
        self.model_type_id = model_type_id
        if self.logs:
           self.logs.write("processing: "+self.uri)    
        if self.logs :
            self.logs.write("configuration time --- seconds ---" + str(time.time() - start_time))
        
        if not test:
            self.process()
        else:
            self.status = 'TestRun'
    
    def process(self):
       # print 'processing recording 1'
        start_time = time.time()
        if not self.hasrec:
            self.instanceRec()
        if self.logs:
            self.logs.write("retrieving recording from storage --- seconds ---" + str(time.time() - start_time))
        #print self.rec.status
        if self.rec.status == 'HasAudioData':
            maxFreqInRec = float(self.rec.sample_rate)/2.0
            if self.high >= maxFreqInRec:
                self.status = 'RoiOutsideRecMaxFreq'
            else:
                if False:#not self.oldModel and float(self.rec.sample_rate) not in analysis_sample_rates:
                    self.status = "SampleRateNotSupported"
                else:
                    start_time = time.time()
                    start_time_all = time.time()
                    self.spectrogram()
                    if self.spec.shape[1] < 2*self.speciesSurface.shape[1]:
                        self.status = 'AudioIsShort'
                        if self.logs:
                           self.logs.write("spectrogrmam --- seconds ---" + str(time.time() - start_time))
                    else:
                        if self.logs:
                            self.logs.write("spectrogrmam --- seconds ---" + str(time.time() - start_time))
                        start_time = time.time()
                        if self.model_type_id == 4:
                            self.featureVector_search()
                        else:
                            self.featureVector()
                        
                        if self.db:
                            elapsed = time.time() - start_time_all
                            #print 'insert into  `recanalizer_stats` (job_id,rec_id,exec_time) VALUES('+str(self.job_id)+','+str(self.rec_id)+','+str(elapsed)+')'
                            with closing(self.db.cursor()) as cursor:
                                cursor.execute("""
                                    INSERT INTO `recanalizer_stats` (job_id, rec_id, exec_time) 
                                    VALUES (%s, %s, %s)
                                """, [
                                    self.job_id, self.rec_id, elapsed
                                ])
                                self.db.commit()                           
                        if self.logs:
                            self.logs.write("Done:feature vector --- seconds ---" + str(time.time() - start_time))
                        self.status = 'Processed'
        else:
            self.status = 'NoData'
        #print self.status
        
    def getRec(self):
        return self.rec
    
    def instanceRec(self):
        self.rec = Rec(str(self.uri),self.tempFolder,self.storage,self.logs,True,False,False)
        self.hasrec = True
        
    def getVector(self ):
        if len(self.distances)<1:
            self.featureVector()
        return self.distances
    
    def insertRecAudio(self,data,fs=44100):
        self.rec = Rec('nouri','/tmp/','storage',None,True,True)
        for i in data:
            self.rec.original.append(i)
        self.rec.sample_rate = fs
        self.rec.resample()
        maxFreqInRec = float(self.rec.sample_rate)/2.0
        self.rec.status = 'HasAudioData'
        self.hasrec = True
        
    def ocurrences(threshold=0.5):
        if len(self.distances)<1:
            self.featureVector()
        if  max(self.distances) < threshold:
            threshold = self.maxDist
    
    def features(self):
        if len(self.distances)<1:
            self.featureVector()
        N = len(self.distances)
        fvi = np.fft.fft(self.distances, n=2*N)
        acf = np.real( np.fft.ifft( fvi * np.conjugate(fvi) )[:N] )
        acf = acf/(N - numpy.arange(N))
        
        xf = abs(numpy.fft.fft(self.distances))

        fs = [ numpy.mean(xf), (max(xf)-min(xf)),
                max(xf), min(xf)
                , numpy.std(xf) , numpy.median(xf),skew(xf),
                kurtosis(xf),acf[0] ,acf[1] ,acf[2]]
        hist = histogram(self.distances,6)[0]
        cfs =  cumfreq(self.distances,6)[0]
        ffs = [fs[0],fs[1],fs[2],fs[3],fs[4],fs[5],fs[6],fs[7],fs[8],fs[9],fs[10]]
        return ffs
    
    def featureVector(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.logs:
               self.logs.write("featureVector start")
            if self.logs:
               self.logs.write(self.uri)    
            pieces = self.uri.split('/')
            self.distances = []
            currColumns = self.spec.shape[1]
            step = self.step
            if self.logs:
               self.logs.write("featureVector start")
            self.matrixSurfacComp = numpy.copy(self.speciesSurface[self.spechigh:self.speclow,:])
           # self.matrixSurfacComp[self.matrixSurfacComp[:,:]==-10000] = numpy.min(self.matrixSurfacComp[self.matrixSurfacComp != -10000])
            winSize = min(self.matrixSurfacComp.shape)
            winSize = min(winSize,7)
            if winSize %2 == 0:
                winSize = winSize - 1
            spec = self.spec;
            #print spec.shape , self.matrixSurfacComp.shape,self.rec.sample_rate
            if self.model_type_id == 3:
                if self.logs:
                    self.logs.write("using search match with zeros")
                self.computeGFTT(numpy.copy(self.matrixSurfacComp),numpy.copy(spec),currColumns)
            elif self.model_type_id == 6:
                if self.logs:
                    self.logs.write("using search match with nans")
                self.computeGFTT_nans(numpy.copy(self.matrixSurfacComp),numpy.copy(spec),currColumns)
            else:
                if self.model_type_id == 1:
                    if self.logs:
                        self.logs.write("using ssim no negatives")
                    for j in range(0,currColumns - self.columns,step):
                        val = ssim( numpy.copy(spec[: , j:(j+self.columns)]) , self.matrixSurfacComp , win_size=winSize)
                        if val < 0:
                           val = 0
                        self.distances.append(  val   )
                if self.model_type_id == 5:
                    if self.logs:
                        self.logs.write("using ssim with negatives")
                    for j in range(0,currColumns - self.columns,step):
                        val = ssim( numpy.copy(spec[: , j:(j+self.columns)]) , self.matrixSurfacComp , win_size=winSize)
                        self.distances.append(  val   )
                if self.model_type_id == 2:
                    if self.logs:
                        self.logs.write("not using ssim - using matrix operations")
                    threshold = Thresholder()
                    matrixSurfacCompCopy = threshold.apply(numpy.copy(self.matrixSurfacComp))
                    specCopy = threshold.apply(numpy.copy(spec))
                    maxnormforsize = numpy.linalg.norm( numpy.ones(shape=matrixSurfacCompCopy.shape) )
                    for j in range(0,currColumns - self.columns,step):
                        val = numpy.linalg.norm( numpy.multiply ( (specCopy[: , j:(j+self.columns)]), matrixSurfacCompCopy ) )/maxnormforsize
                        self.distances.append(  val )
            if self.logs:
               self.logs.write("featureVector end")
 
    def computeGFTT(self,pat,spec,currColumns):
        currColumns = self.spec.shape[1]
        spec = ((spec-numpy.min(numpy.min(spec)))/(numpy.max(numpy.max(spec))-numpy.min(numpy.min(spec))))*255
        spec = spec.astype('uint8')
        self.distances = numpy.zeros(spec.shape[1])
        pat = ((pat-numpy.min(numpy.min(pat)))/(numpy.max(numpy.max(pat))-numpy.min(numpy.min(pat))))*255
        pat = pat.astype('uint8')
        if self.logs:
            self.logs.write("shape:"+str(spec.shape)+''+str(pat.shape)+' '+str(self.distances.shape))
        th, tw = pat.shape[:2]
        result = cv2.matchTemplate(spec, pat, cv2.TM_CCOEFF)
        #self.logs.write(str(result))
        #self.logs.write(str(len(result)))
        #self.logs.write(str(len(result[0])))
        #self.logs.write(str(numpy.max(result)))
        #self.logs.write(str(numpy.mean(result)))
        #self.logs.write(str(numpy.min(result)))
        (_, _, minLoc, maxLoc) = cv2.minMaxLoc(result)
        #self.logs.write(str(minLoc)+' '+str(maxLoc))
        threshold = numpy.percentile(result,98.5)
        loc = numpy.where(result >= threshold)
        winSize = min(pat.shape)
        winSize = min(winSize,7)
        step = 16
        if winSize %2 == 0:
            winSize = winSize - 1
        ssimCalls = 0
        #if self.logs:
               #self.logs.write("searching locations : "+str(len(loc[0])))
               #self.logs.write(str(loc))
        xs = []
        s = -999
        for pt in zip(*loc[::-1]):
            if abs(pt[0] - s)>step/2 :
                xs.append(pt[0])
            s = pt[0]
            
        ##self.logs.write(str(xs))
        xs_smpl = [ xs[i] for i in sorted(random.sample(xrange(len(xs)), min(4,len(xs)))) ]
        #self.logs.write(str(xs_smpl))
        for pts in xs_smpl:
            if pts+math.floor((pat.shape[1]*1.33))<=currColumns:
                for pt in range(max(0,pts-int(math.floor(pat.shape[1]/3))),min(currColumns - self.columns,pts+int(math.floor((pat.shape[1]*1.33)))),step):
                    ssimCalls = ssimCalls + 1
                    val = ssim( numpy.copy(spec[:,pt:(pt+tw)]) , pat, win_size=winSize)
                    if val < 0:
                        val = 0
                    self.distances[pt+tw/2] = val
        # for pt in range(max(0,maxLoc[0]-int(math.floor(pat.shape[1]/3))),min(currColumns - self.columns,maxLoc[0]+int(math.floor((pat.shape[1]*1.33)))),step):
        #     ssimCalls = ssimCalls + 1
        #     val = ssim( numpy.copy(spec[:,pt:(pt+tw)]) , pat, win_size=winSize)
        #     if val < 0:
        #         val = 0
        #     self.distances[pt+tw/2] = val
        if maxLoc[0]+tw<=currColumns:
            ssimCalls = ssimCalls + 1
            val = ssim( numpy.copy(spec[:,maxLoc[0]:(maxLoc[0]+tw)]) , pat, win_size=winSize)
            if val < 0:
                val=0
            self.distances[maxLoc[0]+tw/2] = val
        if self.logs:
         self.logs.write('------------------------- ssimCalls: '+str(ssimCalls)+'-------------------------')
        
    def computeGFTT_nans(self,pat,spec,currColumns):
        currColumns = self.spec.shape[1]
        spec = ((spec-numpy.min(numpy.min(spec)))/(numpy.max(numpy.max(spec))-numpy.min(numpy.min(spec))))*255
        spec = spec.astype('uint8')
        self.distances = []# numpy.zeros(spec.shape[1])
        #self.distances[:] = numpy.NAN
        pat = ((pat-numpy.min(numpy.min(pat)))/(numpy.max(numpy.max(pat))-numpy.min(numpy.min(pat))))*255
        pat = pat.astype('uint8')
        #self.logs.write("shape:"+str(spec.shape)+''+str(pat.shape)+' '+str(self.distances.shape))
        th, tw = pat.shape[:2]
        result = cv2.matchTemplate(spec, pat, cv2.TM_CCOEFF)
        #self.logs.write(str(result))
        #self.logs.write(str(len(result)))
        #self.logs.write(str(len(result[0])))
        #self.logs.write(str(numpy.max(result)))
        #self.logs.write(str(numpy.mean(result)))
        #self.logs.write(str(numpy.min(result)))
        (_, _, minLoc, maxLoc) = cv2.minMaxLoc(result)
        #self.logs.write(str(minLoc)+' '+str(maxLoc))
        threshold = numpy.percentile(result,98.5)
        loc = numpy.where(result >= threshold)
        winSize = min(pat.shape)
        winSize = min(winSize,7)
        step = 16
        if winSize %2 == 0:
            winSize = winSize - 1
        ssimCalls = 0
        #if self.logs:
               #self.logs.write("searching locations : "+str(len(loc[0])))
               #self.logs.write(str(loc))
        xs = []
        s = -999
        for pt in zip(*loc[::-1]):
            if abs(pt[0] - s)>step/2 :
                xs.append(pt[0])
            s = pt[0]
        #self.logs.write(str(xs))
        xs_smpl = [ xs[i] for i in sorted(random.sample(xrange(len(xs)), min(4,len(xs)))) ]
        #self.logs.write(str(xs_smpl))
        for pts in xs_smpl:
            if pts+math.floor((pat.shape[1]*1.33))<=currColumns:
                for pt in range(max(0,pts-int(math.floor(pat.shape[1]/3))),min(currColumns - self.columns,pts+int(math.floor((pat.shape[1]*1.33)))),step):
                    ssimCalls = ssimCalls + 1
                    val = ssim( numpy.copy(spec[:,pt:(pt+tw)]) , pat, win_size=winSize)
                    #self.distances[pt+tw/2] = val
                    self.distances.append( val)
        # for pt in range(max(0,maxLoc[0]-int(math.floor(pat.shape[1]/3))),min(currColumns - self.columns,maxLoc[0]+int(math.floor((pat.shape[1]*1.33)))),step):
        #     ssimCalls = ssimCalls + 1
        #     val = ssim( numpy.copy(spec[:,pt:(pt+tw)]) , pat, win_size=winSize)
        #     if val < 0:
        #         val = 0
        #     self.distances[pt+tw/2] = val
        if maxLoc[0]+tw<=currColumns:
            ssimCalls = ssimCalls + 1
            val = ssim( numpy.copy(spec[:,maxLoc[0]:(maxLoc[0]+tw)]) , pat, win_size=winSize)
            #self.distances[maxLoc[0]+tw/2] = val
            self.distances.append( val)
        if self.logs:
            self.logs.write('------------------------- ssimCalls: '+str(ssimCalls)+'-------------------------')
        
    def featureVector_search(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.logs:
               self.logs.write("featureVector start")
            if self.logs:
               self.logs.write(self.uri)    

            self.matrixSurfacComp = numpy.copy(self.speciesSurface[self.spechigh:self.speclow,:])
            removeUnwanted = self.matrixSurfacComp == -10000
            
           # if len(removeUnwanted) > 0  :
              #  self.matrixSurfacComp[self.matrixSurfacComp[:,:]==-10000] = numpy.min(self.matrixSurfacComp[self.matrixSurfacComp != -10000])
            spec = self.spec
            currColumns = self.spec.shape[1]
            spec = ((spec-numpy.min(numpy.min(spec)))/(numpy.max(numpy.max(spec))-numpy.min(numpy.min(spec))))*255
            spec = spec.astype('uint8')
            pat = self.matrixSurfacComp
            pat = ((pat-numpy.min(numpy.min(pat)))/(numpy.max(numpy.max(pat))-numpy.min(numpy.min(pat))))*255
            pat = pat.astype('uint8')
            th, tw = pat.shape[:2]
            #print pat.shape,spec.shape,self.rec.sample_rate
            result = cv2.matchTemplate(spec, pat, cv2.TM_CCOEFF_NORMED)

            self.distances = numpy.mean(result,axis=0)
            plotAB = False
            if plotAB:
                figure(figsize=(25,15))
                ax1 = subplot(211)
                plot(self.distances)
                subplot(212, sharex=ax1)
                ax = gca()
                im = ax.imshow(spec , interpolation='nearest', aspect='auto')
                savefig(''+self.rec.filename+'mean'+'.png', dpi=100)
                close()

    def getSpec(self):
        return self.spec
    
    def spectrogram(self):
        maxHertzInRec = float(self.rec.sample_rate)/2.0
        i = 0
        nfft = 512
        #if self.rec.sample_rate <= 44100:
        #    while i<len(freqs44100) and freqs44100[i] <= maxHertzInRec :
        #        i = i + 1
        #    nfft = i
        start_time = time.time()

        Pxx, freqs, bins = mlab.specgram(self.rec.original, NFFT=nfft *2, Fs=self.rec.sample_rate , noverlap=nfft )
        #if self.rec.sample_rate < 44100:
        #    self.rec.sample_rate = 44100
        dims =  Pxx.shape
        if self.logs:
            self.logs.write("mlab.specgram --- seconds ---" + str(time.time() - start_time))

        i = 0
        j = 0
        start_time = time.time()
        while freqs[i] < self.low:
            j = j + 1
            i = i + 1
        
        Pxx =  10. * np.log10( Pxx.clip(min=0.0000000001))
        while (i < len(freqs)) and (freqs[i] < self.high):
            i = i + 1
 
        if i >= dims[0]:
            i = dims[0] - 1
            
        if self.model_type_id == 4:
            Z= Pxx[(j-2):(i+2),:]
        else:
            Z= Pxx[(j):(i),:]
        
        self.speclow = len(freqs)-j
        self.spechigh = len(freqs)-i
        Z = np.flipud(Z)
        if self.logs:
            self.logs.write('logs and flip ---' + str(time.time() - start_time))
        self.spec = Z

    
    def showVectAndSpec(self):
        ax1 = subplot(211)
        plot(self.distances)
        subplot(212, sharex=ax1)
        ax = gca()
        im = ax.imshow(self.spec, None)
        ax.axis('auto')
        show()
        close()
        
    def showAudio(self):
        plot(self.rec.original)
        show()
        close()
        
    def showVect(self):
        plot(self.distances)
        show()
        close()
        
    def showspectrogram(self):
        imshow(self.spec)
        show()
        close()
        
    def showSurface(self):
        imshow(self.speciesSurface)
        show()
        close()
 
    def plots(self,s):
       fig, ax = subplots(figsize=(25, 15))
       ax.imshow(s,aspect='auto')
       show()
