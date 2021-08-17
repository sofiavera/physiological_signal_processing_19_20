import inspect
import os

import numpy as np 
import pandas as pd 
from scipy import signal

from scipy import signal
import statistics as st
from sklearn import preprocessing
from sklearn.decomposition import PCA


## señales para leer el set de test--
def read_test(axis,part_body): 
    pwd = os.getcwd()
    path = r'C:/psp-urjc-challenge-1920/Data/Test'

    os.chdir(path)

    subjects = os.listdir()
    sub= np.asarray(subjects,dtype=int)
    sorted_id= np.argsort(sub) #Returns the indices that sort the array.

    subjects_s = sub[sorted_id]
    data = []
    
    if axis == "x":
        i= 0
    if axis == "y":
        i = 1
    if axis == "z":
        i = 2
    
    for s in subjects_s:
        info = np.loadtxt(r"C:psp-urjc-challenge-1920/Data/Test/" + str(subjects_s[s-1]) + "/" + str(part_body) +".txt")[:,i]
        data.append(info)
        
    data= np.asarray(data)
    return data

##fir+ detrend
def prep_test(axis,part_body):
    raw_signal = read_test(axis,part_body)
    fs = 50
    dtrnd_data = []
    prep_data = []
    
    for s in raw_signal: 
        dtrnd_sig = signal.detrend(s)
        dtrnd_data.append(dtrnd_sig)
    
    b = signal.firwin(64,[0.3,5], nyq = fs, pass_zero = False) ### banda de paso [0.3-4]
    for s in dtrnd_data:
        prep_sig = signal.filtfilt(b,1.0,s)
        prep_data.append(prep_sig)
    
    prep_data= np.asarray(prep_data)
    return prep_data

##module
def module_test(part_body): 
    
    x=prep_test("x",part_body)
    y=prep_test("y",part_body)
    
    sg= np.sqrt(np.square(x)+ np.square(y))
    
    return sg

##pca
from sklearn import preprocessing
from sklearn.decomposition import PCA
def pca_test(part_body,n=30): 
    x,y,z, = [], [], []
    xs, ys, zs = [], [], []
    var_rat = np.empty((0,3),int)
    n =  int(n)
    
    valid_patients = []
    
    x = prep_test('x', part_body)[0:n]
    y= prep_test('y', part_body)[0:n]
    z= prep_test('z', part_body)[0:n]
    
    ##standar data
    for i in range(n):
        x_ = preprocessing.scale(x[i])
        y_ = preprocessing.scale(y[i])
        z_ = preprocessing.scale(z[i])
        xs.append(x_)
        ys.append(y_)
        zs.append(z_)
    
    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)
        
   ##apply PCA
    for i in range(n):
        sn = [xs[i],ys[i], zs[i]]
        pca = PCA() 
        pca.fit(sn)
        
        var = pca.explained_variance_ratio_
        var_rat= np.append(var_rat, np.array([var]), axis=0)
        
        #if  (pca.explained_variance_ratio_[0] > 0.75):
        valid_patients.append(pca.components_[0])
    
    return valid_patients

##PSD
def PSD_test(part_body):
    
    PCA = pca_test(part_body)
    fs = 50
    Pxx_list = []
    
    for s in PCA:
        f_welch, Pxx_welch = signal.welch(s,fs=fs, window='hamming',nperseg= 512, nfft = 1024)
        Pxx_list.append(Pxx_welch) 

    return f_welch, Pxx_list

##F0
def F0_test(part_body):
    fs = 50
    n=30
    
    f0_list = []
    x = read_test("x",part_body)
    y = read_test("y",part_body)
    
    for i in range(n):
        x_2 = np.square(x[i])
        y_2 = np.square(y[i])
        
        module = np.sqrt(x_2 + y_2)
    
        b_1,a_1 = signal.butter(4, Wn = 20,btype = "highpass", fs=fs)
        sig_filt1 = signal.filtfilt(b_1,a_1, module)
    
        sig_filt1_abs = np.abs(sig_filt1)
        
        b_2,a_2 = signal.butter(4, Wn = 1.5 ,btype = "lowpass", fs=fs)
        sig_filt2 = signal.filtfilt(b_2,a_2, sig_filt1_abs)

        f_welch, Pxx_welch = signal.welch(sig_filt2, fs = fs, window='hamming',nperseg = 512, nfft = 1024)
        
        f0_index = np.argmax(Pxx_welch)
        f0 = f_welch[f0_index]
        
        f0_list.append(f0)
        
    return f0_list

#TEST PARA Fd (sólo con PCA pq con módulo 0 info)
def Fd_test(part_body):
    n=30
    fd_list = []
    f,a = PSD_test(part_body)
    
    for i in range(n):
        s = a[i]
        
        fd_index = np.argmax(s)
        fd = f[fd_index]
        fd_list.append(fd)

    return fd_list

#TEST PARA INTERVALOS (sólo con módulo pq PCA 0 info)

def test_intervals():
    
    CI = module_test("CI")
    PD = module_test("PD")
    fs = 50
    mIntervals_test = []
    
    for i in np.arange(0,30):
    
        index_CI, _ = signal.find_peaks(CI[i], distance = fs)
        index_PD, _ = signal.find_peaks(PD[i], distance = fs)
        
        index = np.sort(np.concatenate((index_CI, index_PD)))
    
        intervals = []
        
        for i in range(0,len(index)-1):
            if (index[i] in index_CI and index[i+1] in index_PD) or (index[i] in index_PD and index[i+1] in index_CI):
                intervals.append((index[i+1]- index[i])/fs)   

        standd_d = st.stdev(intervals)
        mIntervals_test.append(standd_d)
        
    return mIntervals_test

####para power con modulo### [0.3-15]

def prep_m_test(axis, part_body):
    raw_signal = read_test(axis, part_body)
    fs = 50
    dtrnd_data = []
    prep_data = []
    
    for s in raw_signal: 
        dtrnd_sig = signal.detrend(s)
        dtrnd_data.append(dtrnd_sig)
    
    b = signal.firwin(64,[0.3,15], nyq = fs, pass_zero = False)
    for s in dtrnd_data:
        prep_sig = signal.filtfilt(b,1.0,s)
        
        prep_data.append(prep_sig)
        
    prep_data= np.asarray(prep_data)
    
    return prep_data

def module_m_test(part_body):
    
    x= prep_m_test("x",part_body)
    y= prep_m_test("y",part_body)
    
    s= np.sqrt(np.square(x)+ np.square(y))   
    
    return s


def power_m_test(part_body):
    
    fs=50
    totalp, bandp = [], []
    m = module_m_test(part_body)
    
    ## total average power of the signal:
    for s in m:        
        f, P= signal.welch(s,fs=fs, window='hamming',nperseg =512, nfft= 1024)
        width = np.diff(f)
        pvec_total= np.multiply(width,P[0:-1])
        avgp_total = np.sum(pvec_total)
        #poner argumentos en la función 
        #print('total average power signal', s +1,': ', avgp_total)
        totalp.append(avgp_total)
        
        ## absolute band power
        if part_body== 'CI': ###DE MOMENTO DEJAR IGUAL
            a= 1
            b= 3
        else:
            a=1
            b=3
    
        f_low_index = (np.abs(f-a)).argmin()   
        f_high_index = (np.abs(f-b)).argmin() 
    
        pvec_= np.multiply(width[f_low_index:f_high_index],P[f_low_index:f_high_index])
        avgp_ = np.sum(pvec_) 
        bandp.append(avgp_)
    
    
    bandp = np.asarray(bandp)
    totalp = np.asarray(totalp)
    
    ri = bandp/totalp
    return  ri , totalp, bandp 



##power test PCA
def power_test(part_body):    
    fs=50
    totalp, bandp = [], [] 
    PCA = pca_test(part_body)
    
    ## total average power of the signal
    for s in PCA:
        f, P= signal.welch(s,fs=fs, window='hamming', nperseg = 512, nfft =1024)
        width = np.diff(f)
        pvec_total= np.multiply(width,P[0:-1])
        avgp_total = np.sum(pvec_total)        
        totalp.append(avgp_total)
    
        ## absolute band power
        if part_body== 'CI': ###DE MOMENTO DEJAR IGUAL
            a= 1
            b= 3
        else:
            a=1
            b=3
            
        f_low_index = (np.abs(f-a)).argmin()   
        f_high_index = (np.abs(f-b)).argmin() 
    
        pvec_= np.multiply(width[f_low_index:f_high_index],P[f_low_index:f_high_index])
        avgp_ = np.sum(pvec_) 
        bandp.append(avgp_)
        
    
    totalp= np.asarray(totalp)
    bandp= np.asarray(bandp)
        
    ## relative band power
    ri = bandp/totalp
    return  ri , totalp, bandp  


# function to write the definition of our function to the file
def write_function_to_file(function, file):
    if os.path.exists(file):
        append_write = 'a' # append if already exists
    else:
        append_write = 'w' # make a new file if not
    with open(file, append_write) as file:
        function_definition = inspect.getsource(function)
        file.write(function_definition)

# write both of our functions to our output file        
write_function_to_file(read_test, "test.py")
write_function_to_file(prep_test, "test.py")
write_function_to_file(module_test, "test.py")
write_function_to_file(pca_test, "test.py")
write_function_to_file(PSD_test, "test.py")
write_function_to_file(F0_test, "test.py")
write_function_to_file(Fd_test, "test.py")
write_function_to_file(test_intervals, "test.py")
write_function_to_file(power_test, "test.py")
write_function_to_file(prep_m_test, "test.py")
write_function_to_file(module_m_test, "test.py")
write_function_to_file(power_m_test, "test.py")






