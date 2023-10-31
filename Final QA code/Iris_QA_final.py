# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 15:06:41 2023

@author: SchockSinjaTheresa
"""

import numpy as np
import matplotlib.pyplot as plt
from helper_final import file_einlesen, get_nearest, interpolate, Flatness_siemens, get_profiles, symmetry
import os

#IrisQA Daten sollten aus 12 Files bestehen, eines pro Kollimatorgröße, beginnend mit 5mm, endend mit 60mm
path = 'Data examples/IrisQA example'
files = os.listdir(path)
n = len(files)


i=0
FW_x = np.zeros(n)
FW_y = np.zeros(n)
flat_x = np.zeros(n)
flat_y = np.zeros(n)
sym_x = np.zeros(n)
sym_y = np.zeros(n)
center = np.empty(2)

for file in files:
    j = (i)%12
    data, x,y,_ = file_einlesen(path+'/'+str(file))
    
    profiles, center_val, center, half_dose = get_profiles(data, x, y, cax_correction = True)  
    #profiles are evaluated at closest pixel position to calculated center, center is pixel index

    a=np.empty(2)
    b = np.empty(2)
    
    [left_x, right_x,a[0] ,b[0]] = interpolate(profiles[0], x, half_dose, integrate=True, c = center[0]) #entweder Field Center oder mean (0.57, 1.09)
    [left_y, right_y, a[1] ,b[1]] = interpolate(profiles[1], y, half_dose, integrate=True, c= center[1])
    FW_x[j] = right_x-left_x
    FW_y[j] = right_y-left_y
    center_val = [left_x+right_x, left_y+right_y]
    
    '''for Flatness calculate percentage dose difference according to IEC:
        (Dmax-Dmin)/(Dmax+Dmin)*100 in flattened region on central profiles
        flattened region defined as 80% of field width from center of beam (c_x,c_y)
        This is due to the shift of the beams in relation to central axis,
        which matters especially for small fieldwidths'''
    Flat_siem_x, Flat_PTW_x = Flatness_siemens(profiles[0], FW_x[j], center[0], x) #two different definitions are available to compare data also with Octavius 1000SRS PTW measurements
    Flat_siem_y, Flat_PTW_y = Flatness_siemens(profiles[1], FW_y[j], center[1], y)
    [flat_x[j], flat_y[j]] = [Flat_siem_x, Flat_siem_y]
    
    [sym_x[j], sym_y[j]] = symmetry(a,b)
    
    print('Field size in x, y, beam center')
    print(FW_x[j],FW_y[j], np.array(center_val).flatten()/2)
    i+=1
    
    
print('flatness:')
print(flat_x)
print(flat_y)
print('symmetry')
print(sym_x)
print(sym_y)
    
#%% Test with reference values
FW_x_ref = np.array([5.049, 7.759, 10.196, 12.590, 15.072, 20.059, 25.007, 29.97, 34.78, 39.74, 49.71, 59.71])
FW_y_ref = np.array([5.21, 7.907, 10.381, 12.787, 15.288, 20.297, 25.266, 30.26, 35.10, 40.08, 50.11, 60.16])

colli = np.array([5,7.5,10,12.5,15,20,25,30,35,40,50,60])

dif_x = np.abs(FW_x_ref - FW_x)
dif_y = np.abs(FW_y_ref - FW_y)



if np.all(np.array([dif_x, dif_y])<0.2):
    print('All field sizes within tolerance')
else:
    fail_x = np.argwhere(dif_x>=0.2)
    fail_y = np.argwhere(dif_y>=0.2)
    print('failed beam:', colli[fail_x], colli[fail_y])
    
    
    
