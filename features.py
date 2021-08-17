import inspect
import os

import numpy as np 
import pandas as pd
from scipy import signal
import statistics as st

import pre_processing as pr
import power_spectral_density as psd

###Fundamental Frequency

def F0(type_patient, part_body):
    fs = 50
    n=20
    
    f0_list = []
    x = pr.read_data(type_patient,"x",part_body)
    y = pr.read_data(type_patient,"y",part_body)
    
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

### dominant frequency
def mFd(type_patient, part_body):
    n=20
    fd_list = []
    f,a = psd.mPSD(type_patient, part_body)
    
    for i in range(n):
        s = a[i]
        
        fd_index = np.argmax(s)
        fd = f[fd_index]
        fd_list.append(fd)

    return fd_list

##con pca
def pFd(type_patient, part_body):
    n=20
    fd_list = []
    f,a = psd.pPSD(type_patient, part_body)
    
    for i in range(n):
        s = a[i]
        
        fd_index = np.argmax(s)
        fd = f[fd_index]
        fd_list.append(fd)

    return fd_list

## intervals with the module
def mIntervals(type_patient):
    
    fs =50

    CI = pr.module(type_patient,"CI")
    PD = pr.module(type_patient,"PD")
    
    standd_d_list = []
    
    for i in np.arange(0,20):
    
        index_CI, _ = signal.find_peaks(CI[i], distance = fs)
        index_PD, _ = signal.find_peaks(PD[i], distance = fs)
        
        index = np.sort(np.concatenate((index_CI, index_PD)))
        intervals = []
        
        for i in range(0,len(index)-1):
            if (index[i] in index_CI and index[i+1] in index_PD) or (index[i] in index_PD and index[i+1] in index_CI):
                intervals.append((index[i+1]- index[i])/fs)   

        standd_d = st.stdev(intervals)
        standd_d_list.append(standd_d)
    
    return standd_d_list

##intervals with  pca
def pIntervals(type_patient):

    CI = pr.pca_method(type_patient,"CI")
    PD = pr.pca_method(type_patient,"PD")
    
    standd_d_list = []
    
    for i in np.arange(0,20):
    
        index_CI, _ = signal.find_peaks(CI[i], distance = fs)
        index_PD, _ = signal.find_peaks(PD[i], distance = fs)
        
        index = np.sort(np.concatenate((index_CI, index_PD)))
        
        intervals = []
        
        for i in range(0,len(index)-1):
            if (index[i] in index_CI and index[i+1] in index_PD) or (index[i] in index_PD and index[i+1] in index_CI):
                intervals.append((index[i+1]- index[i])/fs)   

        standd_d = st.stdev(intervals)
        standd_d_list.append(standd_d)
    
    return standd_d_list

###power pca
def power(type_patient, part_body): 
    
    fs=50
    totalp, bandp, ri = [], [], []
    PCA = pr.pca_method(type_patient,part_body)
    
    
    for s in PCA:        
        f, P= signal.welch(s,fs=fs, window='hamming',nperseg =512, nfft= 1024)
        ## total average power of the signal:
        width = np.diff(f)
        pvec_total= np.multiply(width,P[0:-1])
        avgp_total = np.sum(pvec_total)
        totalp.append(avgp_total)

        ## absolute band power
        if part_body== 'CI': ###probar si mejora cambiando la banda de frecuencia
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
    
    ##relative band power
    ri = bandp/totalp
    return  ri , totalp, bandp

#### power module full band
def prep_m(type_patient,axis,part_body):
    raw_signal = pr.read_data(type_patient,axis,part_body)
    fs = 50
    dtrnd_data = []
    prep_data = []
    
    for s in raw_signal: 
        dtrnd_sig = signal.detrend(s)
        dtrnd_data.append(dtrnd_sig)
    
    b = signal.firwin(64,[0.3,15], nyq = fs, pass_zero = False) ###preguntar a SOFI
    for s in dtrnd_data:
        prep_sig = signal.filtfilt(b,1.0,s)
        
        prep_data.append(prep_sig)
        
    prep_data= np.asarray(prep_data)
    
    return prep_data

def module_m(type_patient,part_body):
    
    x= prep_m(type_patient,"x",part_body)
    y= prep_m(type_patient,"y",part_body)
    
    s= np.sqrt(np.square(x)+ np.square(y))   
    
    return s

def power_m(type_patient, part_body):
    
    fs=50
    totalp, bandp = [], []
    m = module_m(type_patient,part_body)
   
    for s in m:       
        ## total average power of the signal:
        f, P= signal.welch(s,fs=fs, window='hamming',nperseg =512, nfft= 1024)
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
    
    
    bandp = np.asarray(bandp)
    totalp = np.asarray(totalp)
    
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

write_function_to_file(F0, "features.py")
write_function_to_file(mFd, "features.py")
write_function_to_file(pFd, "features.py")
write_function_to_file(mIntervals, "features.py")
write_function_to_file(pIntervals, "features.py")
write_function_to_file(power, "features.py")
write_function_to_file(prep_m, "features.py")
write_function_to_file(module_m, "features.py")
write_function_to_file(power_m, "features.py")




def F0(type_patient, part_body):
    fs = 50
    n=20
    
    f0_list = []
    x = pr.read_data(type_patient,"x",part_body)
    y = pr.read_data(type_patient,"y",part_body)
    
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
def mFd(type_patient, part_body):
    n=20
    fd_list = []
    f,a = psd.mPSD(type_patient, part_body)
    
    for i in range(n):
        s = a[i]
        
        fd_index = np.argmax(s)
        fd = f[fd_index]
        fd_list.append(fd)

    return fd_list
def pFd(type_patient, part_body):
    n=20
    fd_list = []
    f,a = psd.pPSD(type_patient, part_body)
    
    for i in range(n):
        s = a[i]
        
        fd_index = np.argmax(s)
        fd = f[fd_index]
        fd_list.append(fd)

    return fd_list
def mIntervals(type_patient):
    
    fs =50

    CI = pr.module(type_patient,"CI")
    PD = pr.module(type_patient,"PD")
    
    standd_d_list = []
    
    for i in np.arange(0,20):
    
        index_CI, _ = signal.find_peaks(CI[i], distance = fs)
        index_PD, _ = signal.find_peaks(PD[i], distance = fs)
        
        index = np.sort(np.concatenate((index_CI, index_PD)))
        intervals = []
        
        for i in range(0,len(index)-1):
            if (index[i] in index_CI and index[i+1] in index_PD) or (index[i] in index_PD and index[i+1] in index_CI):
                intervals.append((index[i+1]- index[i])/fs)   

        standd_d = st.stdev(intervals)
        standd_d_list.append(standd_d)
    
    return standd_d_list
def pIntervals(type_patient):

    CI = pr.pca_method(type_patient,"CI")
    PD = pr.pca_method(type_patient,"PD")
    
    standd_d_list = []
    
    for i in np.arange(0,20):
    
        index_CI, _ = signal.find_peaks(CI[i], distance = fs)
        index_PD, _ = signal.find_peaks(PD[i], distance = fs)
        
        index = np.sort(np.concatenate((index_CI, index_PD)))
        
        intervals = []
        
        for i in range(0,len(index)-1):
            if (index[i] in index_CI and index[i+1] in index_PD) or (index[i] in index_PD and index[i+1] in index_CI):
                intervals.append((index[i+1]- index[i])/fs)   

        standd_d = st.stdev(intervals)
        standd_d_list.append(standd_d)
    
    return standd_d_list
def power(type_patient, part_body): 
    
    fs=50
    totalp, bandp, ri = [], [], []
    PCA = pr.pca_method(type_patient,part_body)
    
    
    for s in PCA:        
        f, P= signal.welch(s,fs=fs, window='hamming',nperseg =512, nfft= 1024)
        ## total average power of the signal:
        width = np.diff(f)
        pvec_total= np.multiply(width,P[0:-1])
        avgp_total = np.sum(pvec_total)
        totalp.append(avgp_total)

        ## absolute band power
        if part_body== 'CI': ###probar si mejora cambiando la banda de frecuencia
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
    
    ##relative band power
    ri = bandp/totalp
    return  ri , totalp, bandp
def prep_m(type_patient,axis,part_body):
    raw_signal = pr.read_data(type_patient,axis,part_body)
    fs = 50
    dtrnd_data = []
    prep_data = []
    
    for s in raw_signal: 
        dtrnd_sig = signal.detrend(s)
        dtrnd_data.append(dtrnd_sig)
    
    b = signal.firwin(64,[0.3,15], nyq = fs, pass_zero = False) ###preguntar a SOFI
    for s in dtrnd_data:
        prep_sig = signal.filtfilt(b,1.0,s)
        
        prep_data.append(prep_sig)
        
    prep_data= np.asarray(prep_data)
    
    return prep_data
def module_m(type_patient,part_body):
    
    x= prep_m(type_patient,"x",part_body)
    y= prep_m(type_patient,"y",part_body)
    
    s= np.sqrt(np.square(x)+ np.square(y))   
    
    return s
def power_m(type_patient, part_body):
    
    fs=50
    totalp, bandp = [], []
    m = module_m(type_patient,part_body)
   
    for s in m:       
        ## total average power of the signal:
        f, P= signal.welch(s,fs=fs, window='hamming',nperseg =512, nfft= 1024)
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
    
    
    bandp = np.asarray(bandp)
    totalp = np.asarray(totalp)
    
    ri = bandp/totalp
    return  ri , totalp, bandp 
def F0(type_patient, part_body):
    fs = 50
    n=20
    
    f0_list = []
    x = pr.read_data(type_patient,"x",part_body)
    y = pr.read_data(type_patient,"y",part_body)
    
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
def mFd(type_patient, part_body):
    n=20
    fd_list = []
    f,a = psd.mPSD(type_patient, part_body)
    
    for i in range(n):
        s = a[i]
        
        fd_index = np.argmax(s)
        fd = f[fd_index]
        fd_list.append(fd)

    return fd_list
def pFd(type_patient, part_body):
    n=20
    fd_list = []
    f,a = psd.pPSD(type_patient, part_body)
    
    for i in range(n):
        s = a[i]
        
        fd_index = np.argmax(s)
        fd = f[fd_index]
        fd_list.append(fd)

    return fd_list
def mIntervals(type_patient):
    
    fs =50

    CI = pr.module(type_patient,"CI")
    PD = pr.module(type_patient,"PD")
    
    standd_d_list = []
    
    for i in np.arange(0,20):
    
        index_CI, _ = signal.find_peaks(CI[i], distance = fs)
        index_PD, _ = signal.find_peaks(PD[i], distance = fs)
        
        index = np.sort(np.concatenate((index_CI, index_PD)))
        intervals = []
        
        for i in range(0,len(index)-1):
            if (index[i] in index_CI and index[i+1] in index_PD) or (index[i] in index_PD and index[i+1] in index_CI):
                intervals.append((index[i+1]- index[i])/fs)   

        standd_d = st.stdev(intervals)
        standd_d_list.append(standd_d)
    
    return standd_d_list
def pIntervals(type_patient):

    CI = pr.pca_method(type_patient,"CI")
    PD = pr.pca_method(type_patient,"PD")
    
    standd_d_list = []
    
    for i in np.arange(0,20):
    
        index_CI, _ = signal.find_peaks(CI[i], distance = fs)
        index_PD, _ = signal.find_peaks(PD[i], distance = fs)
        
        index = np.sort(np.concatenate((index_CI, index_PD)))
        
        intervals = []
        
        for i in range(0,len(index)-1):
            if (index[i] in index_CI and index[i+1] in index_PD) or (index[i] in index_PD and index[i+1] in index_CI):
                intervals.append((index[i+1]- index[i])/fs)   

        standd_d = st.stdev(intervals)
        standd_d_list.append(standd_d)
    
    return standd_d_list
def power(type_patient, part_body): 
    
    fs=50
    totalp, bandp, ri = [], [], []
    PCA = pr.pca_method(type_patient,part_body)
    
    
    for s in PCA:        
        f, P= signal.welch(s,fs=fs, window='hamming',nperseg =512, nfft= 1024)
        ## total average power of the signal:
        width = np.diff(f)
        pvec_total= np.multiply(width,P[0:-1])
        avgp_total = np.sum(pvec_total)
        totalp.append(avgp_total)

        ## absolute band power
        if part_body== 'CI': ###probar si mejora cambiando la banda de frecuencia
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
    
    ##relative band power
    ri = bandp/totalp
    return  ri , totalp, bandp
def prep_m(type_patient,axis,part_body):
    raw_signal = pr.read_data(type_patient,axis,part_body)
    fs = 50
    dtrnd_data = []
    prep_data = []
    
    for s in raw_signal: 
        dtrnd_sig = signal.detrend(s)
        dtrnd_data.append(dtrnd_sig)
    
    b = signal.firwin(64,[0.3,15], nyq = fs, pass_zero = False) ###preguntar a SOFI
    for s in dtrnd_data:
        prep_sig = signal.filtfilt(b,1.0,s)
        
        prep_data.append(prep_sig)
        
    prep_data= np.asarray(prep_data)
    
    return prep_data
def module_m(type_patient,part_body):
    
    x= prep_m(type_patient,"x",part_body)
    y= prep_m(type_patient,"y",part_body)
    
    s= np.sqrt(np.square(x)+ np.square(y))   
    
    return s
def power_m(type_patient, part_body):
    
    fs=50
    totalp, bandp = [], []
    m = module_m(type_patient,part_body)
   
    for s in m:       
        ## total average power of the signal:
        f, P= signal.welch(s,fs=fs, window='hamming',nperseg =512, nfft= 1024)
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
    
    
    bandp = np.asarray(bandp)
    totalp = np.asarray(totalp)
    
    ri = bandp/totalp
    return  ri , totalp, bandp 
