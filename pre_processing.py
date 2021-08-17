### PREPOCESSING: FILTERING AND DIMENSIONALITY REDUCTION
import inspect
import os

import numpy as np 
import pandas as pd
from scipy import signal

import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.decomposition import PCA

###read data set
def read_data(type_patient,axis,part_body):  
    train = pd.read_csv(r'C:psp-urjc-challenge-1920/Training.csv')
    is_group = train['Category'] == int(type_patient)
    G = train[is_group]
    
    data = []
    fs = 50
    
    if axis == "x":
        i= 0
    if axis == "y":
        i = 1
    if axis == "z":
        i = 2
    
    for index,row in G.iterrows():
        info = np.loadtxt(r"C:psp-urjc-challenge-1920/Data/Training/"+ str(row['Id']) + "/" + part_body + ".txt")[:,i]
        data.append(info)
        
    return data

## FIR filter [0.3,5]
def filtering(type_patient,axis,part_body):
    raw_signal = read_data(type_patient,axis,part_body)
    fs = 50
    dtrnd_data = []
    prep_data = []
    
    for s in raw_signal: 
        dtrnd_sig = signal.detrend(s)
        dtrnd_data.append(dtrnd_sig)
    
    b = signal.firwin(64,[0.3,5], nyq = fs, pass_zero = False) 
    for s in dtrnd_data:
        prep_sig = signal.filtfilt(b,1.0,s)
        prep_data.append(prep_sig)
    
    return prep_data

### Module
def module(type_patient,part_body):
    sg = []
    
    x= filtering(type_patient,"x",part_body)
    y= filtering(type_patient,"y",part_body)
    
    s= np.sqrt(np.square(x)+ np.square(y))
    sg.append(s)    
    
    return sg[0]

###pca
def pca_method(type_patient,part_body,n=20, plot = False, explained_variance_ratio_all = False, show_all = False, mean_var_g = False): 
    x,y,z, = [], [], []
    xs, ys, zs = [], [], []
    var_rat = np.empty((0,3),int)
    n =  int(n)
    valid_patients = []
    
    x = filtering(type_patient, 'x', part_body)[0:n]
    y= filtering(type_patient, 'y', part_body)[0:n]
    z= filtering(type_patient, 'z', part_body)[0:n]
    
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
        
        if explained_variance_ratio_all == True and show_all == True:
            print('patient', i+1 ,'result:', pca.explained_variance_ratio_)
        
        if plot == True and show_all == True:
            plt.figure(figsize = (10,5))
            plt.plot(pca.components_.T)
            plt.xlim(100,200)
            plt.ylim(-0.025,0.025)
            plt.legend(['First component','Second component','Third component'])
    
    if explained_variance_ratio_all== True and show_all == False:
            print('patient', i+1 ,'explained variance ratio:', pca.explained_variance_ratio_)
        
    
    if plot == True and show_all == False:
            plt.figure(figsize = (10,5))
            plt.plot(pca.components_.T) ### me plotearÃ¡ el Ãºltimo que es el que quiero 
            plt.xlim(100,200)
            plt.ylim(-0.025,0.025)
            plt.legend(['First component','Second component','Third component'])

 
    if n == 20 and mean_var_g == True:
        print('mean value of the components for',part_body,'from group', type_patient, var_rat.mean(axis=0)) 
        
    return valid_patients

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
write_function_to_file(read_data, "pre_processing.py")
write_function_to_file(filtering, "pre_processing.py")
write_function_to_file(module, "pre_processing.py")
write_function_to_file(pca_method, "pre_processing.py")
def read_data(type_patient,axis,part_body):  
    train = pd.read_csv(r'C:psp-urjc-challenge-1920/Training.csv')
    is_group = train['Category'] == int(type_patient)
    G = train[is_group]
    
    data = []
    fs = 50
    
    if axis == "x":
        i= 0
    if axis == "y":
        i = 1
    if axis == "z":
        i = 2
    
    for index,row in G.iterrows():
        info = np.loadtxt(r"C:psp-urjc-challenge-1920/Data/Training/"+ str(row['Id']) + "/" + part_body + ".txt")[:,i]
        data.append(info)
        
    return data
def filtering(type_patient,axis,part_body):
    raw_signal = read_data(type_patient,axis,part_body)
    fs = 50
    dtrnd_data = []
    prep_data = []
    
    for s in raw_signal: 
        dtrnd_sig = signal.detrend(s)
        dtrnd_data.append(dtrnd_sig)
    
    b = signal.firwin(64,[0.3,5], nyq = fs, pass_zero = False) 
    for s in dtrnd_data:
        prep_sig = signal.filtfilt(b,1.0,s)
        prep_data.append(prep_sig)
    
    return prep_data
def module(type_patient,part_body):
    sg = []
    
    x= filtering(type_patient,"x",part_body)
    y= filtering(type_patient,"y",part_body)
    
    s= np.sqrt(np.square(x)+ np.square(y))
    sg.append(s)    
    
    return sg[0]
def pca_method(type_patient,part_body,n=20, plot = False, explained_variance_ratio_all = False, show_all = False, mean_var_g = False): 
    x,y,z, = [], [], []
    xs, ys, zs = [], [], []
    var_rat = np.empty((0,3),int)
    n =  int(n)
    valid_patients = []
    
    x = filtering(type_patient, 'x', part_body)[0:n]
    y= filtering(type_patient, 'y', part_body)[0:n]
    z= filtering(type_patient, 'z', part_body)[0:n]
    
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
        
        if explained_variance_ratio_all == True and show_all == True:
            print('patient', i+1 ,'result:', pca.explained_variance_ratio_)
        
        if plot == True and show_all == True:
            plt.figure(figsize = (10,5))
            plt.plot(pca.components_.T)
            plt.xlim(100,200)
            plt.ylim(-0.025,0.025)
            plt.legend(['First component','Second component','Third component'])
    
    if explained_variance_ratio_all== True and show_all == False:
            print('patient', i+1 ,'explained variance ratio:', pca.explained_variance_ratio_)
        
    
    if plot == True and show_all == False:
            plt.figure(figsize = (10,5))
            plt.plot(pca.components_.T) ### me ploteará el último que es el que quiero 
            plt.xlim(100,200)
            plt.ylim(-0.025,0.025)
            plt.legend(['First component','Second component','Third component'])

 
    if n == 20 and mean_var_g == True:
        print('mean value of the components for',part_body,'from group', type_patient, var_rat.mean(axis=0)) 
        
    return valid_patients
