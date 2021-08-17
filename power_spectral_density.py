import inspect
import os

import numpy as np 
import pandas as pd
from scipy import signal

import pre_processing as pr

##periodograma para el modulo
def mPSD(type_patient,part_body):
    
    mod = pr.module(type_patient,part_body)
    
    fs = 50
    Pxx_list = []
    
    for s in mod:
        f_welch, Pxx_welch = signal.welch(s,fs=fs, window='hamming',nperseg= 512, nfft = 1024)
        Pxx_list.append(Pxx_welch) 

    return f_welch, Pxx_list

##periodograma con pca
def pPSD(type_patient,part_body):
    
    PCA = pr.pca_method(type_patient,part_body)
    fs = 50
    Pxx_list = []
    
    for s in PCA:
        f_welch, Pxx_welch = signal.welch(s,fs=fs, window='hamming',nperseg= 512, nfft = 1024)
        Pxx_list.append(Pxx_welch) 

    return f_welch, Pxx_list

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
write_function_to_file(mPSD, "power_spectral_density.py")
write_function_to_file(pPSD, "power_spectral_density.py")


def mPSD(type_patient,part_body):
    
    mod = pr.module(type_patient,part_body)
    
    fs = 50
    Pxx_list = []
    
    for s in mod:
        f_welch, Pxx_welch = signal.welch(s,fs=fs, window='hamming',nperseg= 512, nfft = 1024)
        Pxx_list.append(Pxx_welch) 

    return f_welch, Pxx_list
def pPSD(type_patient,part_body):
    
    PCA = pr.pca_method(type_patient,part_body)
    fs = 50
    Pxx_list = []
    
    for s in PCA:
        f_welch, Pxx_welch = signal.welch(s,fs=fs, window='hamming',nperseg= 512, nfft = 1024)
        Pxx_list.append(Pxx_welch) 

    return f_welch, Pxx_list
def mPSD(type_patient,part_body):
    
    mod = pr.module(type_patient,part_body)
    
    fs = 50
    Pxx_list = []
    
    for s in mod:
        f_welch, Pxx_welch = signal.welch(s,fs=fs, window='hamming',nperseg= 512, nfft = 1024)
        Pxx_list.append(Pxx_welch) 

    return f_welch, Pxx_list
def pPSD(type_patient,part_body):
    
    PCA = pr.pca_method(type_patient,part_body)
    fs = 50
    Pxx_list = []
    
    for s in PCA:
        f_welch, Pxx_welch = signal.welch(s,fs=fs, window='hamming',nperseg= 512, nfft = 1024)
        Pxx_list.append(Pxx_welch) 

    return f_welch, Pxx_list
