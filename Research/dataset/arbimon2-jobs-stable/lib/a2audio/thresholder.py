try:
    from skimage.filters import threshold_otsu, threshold_adaptive, threshold_isodata, threshold_yen
except:
    from skimage.filter import threshold_otsu, threshold_adaptive, threshold_isodata, threshold_yen
import numpy
from sklearn.cluster import k_means

funcs = {'global':{'otsu','median','kmeans','isodata','yen'} ,
         'adaptive':{ 'gaussian', 'mean', 'median'}}

class Thresholder:
        
    def __init__(self , func = 'adaptive' , method = 'mean'):
        
        if func in funcs:
            self.func = func
            if method in funcs[func]:
                self.method = method
            else:
                raise ValueError('unkown method '+str(method)+' for func '+str(func))
        else:
            raise ValueError('unkown func '+str(func))      
                
    def apply(self,matrix):
        binary = []
        
        if self.func == 'global':
            value = 0
            if self.method == 'otsu':
                value = threshold_otsu(matrix)
            if self.method == 'isodata':
                value = threshold_isodata(matrix)
            if self.method == 'yen':
                value = threshold_yen(matrix)
            if self.method == 'median':
                value = numpy.median(matrix)
            if self.method == 'kmeans':
                aa = numpy.array(matrix).reshape(-1)
                aa.shape = (aa.shape[0],1)
                cc = k_means(aa,5)
                ccc = cc[0].reshape(-1)
                ccc.sort()
                value = ccc[len(ccc)-2]
            binary = matrix > value
            
        if  self.func ==  'adaptive':
            binary = threshold_adaptive(matrix, 127, self.method)
            
        return binary.astype('float')

