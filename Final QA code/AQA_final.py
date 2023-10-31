# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 14:40:08 2022

@author: SchockSinjaTheresa
"""

#test

import numpy as np
from helper_final import file_einlesen, get_nearest, interpolate
import matplotlib.pyplot as plt
import os


path = 'D:\Masterarbeit\Daten\Github\AQA example'
collimator = 'fixed'  #Wähle 'fixed', 'Iris' oder 'MLC'
files = os.listdir(path)
n = len(files)

i=0
python_x = np.zeros(n)
python_y = np.zeros(n)
center=[]


for file in files:
    data, x,y,_ = file_einlesen(path+'/'+str(file))
    
    
    
    '''read in profiles in x and y at 0 with linear interpolation'''
    ind_y_0 = np.argwhere(y==0.2)
    ind_y_1 = np.argwhere(y==-0.2)
    ind_x_0 = np.argwhere(x==0.2)
    ind_x_1 = np.argwhere(x==-0.2)
    profile_x = (data[:, ind_y_0] + data[:, ind_y_1])/2 #linear interpolation for profile extraction
    profile_x = profile_x.reshape(300)
    profile_y = (data[ind_x_0, :] + data[ind_x_1, :])/2 #linear interpolation for profile extraction
    profile_y = np.array(profile_y.reshape(350))
    
    cax_dose = (data[ind_x_0, ind_y_0]+data[ind_x_0, ind_y_1] + 
                data[ind_x_1,ind_y_0] + data[ind_x_1,ind_y_1]).squeeze(0)/4
    half_dose = 0.5*cax_dose
    

    
    '''a) in x: determine field size at 50% CAX-Dose with interpolation'''
    
    [left, right,_ ,_] = interpolate(profile_x, x, half_dose, integrate=True)
    Field_center_x = (left+right)/2
    field_size_x = right-left
    _, c_x = get_nearest(x,Field_center_x,1)
    python_x[i] = field_size_x


    '''b) in y'''
    [left, right, _ , _] = interpolate(profile_y, y, half_dose, integrate=True)
    field_size_y = right-left
    Field_center_y = (left+right)/2
    _, c_y = get_nearest(y,Field_center_y,1)
    python_y[i] = field_size_y

    
    '''b.2) center correction'''
    profile_y = data[c_x, :].reshape(350)
    profile_x = data[:, c_y].reshape(300)
    cax_dose = data[c_x, c_y]
    half_dose = 0.5*cax_dose
    
    [left, right,_ , _] = interpolate(profile_x, x, half_dose, integrate=True)
    Field_center_x = (left+right)/2
    field_size_x = right-left
    python_x[i] = field_size_x
    
    [left, right, _ , _] = interpolate(profile_y, y, half_dose, integrate=True)
    field_size_y = right-left
    Field_center_y = (left+right)/2
    python_y[i] = field_size_y
    center.append([Field_center_x, Field_center_y])

    i+=1



x = np.array(center)[::2,0].flatten()/np.sqrt(2)  #in x und z muss jeweils um factor sqrt(2) korrigiert werden, um von gemessener verschiebung auf tatsächliche verschiebung zu schließen
y = (np.array(center)[::2,1]+np.array(center)[1::2,1]).flatten()/2 #in y wird Mittelwert aus beiden Messungen gebildet
z = np.array(center)[1::2,0].flatten()/np.sqrt(2)


#%% Compare with baseline values
if collimator == 'fixed':
    z_mean = 0.82
    z_std = 0.08
    y_mean = -2.25
    y_std = 0.04
    x_mean = 1.05
    x_std = 0.11
elif collimator == 'Iris':
    z_mean = 0.67
    z_std = 0.06
    y_mean = -2.33
    y_std = 0.04
    x_mean = 1.21
    x_std = 0.09
elif collimator =='MLC':
    z_mean = 0.62
    z_std = 0.03
    y_mean = -10.13
    y_std = 0.05
    x_mean = 0.91
    x_std = 0.13

mean = [x_mean, y_mean, z_mean]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(0,0,0, label='mean')
dev = []
c = 'purple'
for l in range(int(n/2)):

    #print(l, l%11)
    xp = x[l]
    yp = y[l]
    zp = z[l]
    dist = np.linalg.norm(np.array(mean)-np.array([xp, yp, zp]))
    if dist >0.65:
        c = 'red'
        label = 'AQA failed'
    else:
        c = 'purple'
        label = 'AQA passed'
        print('AQA passed')
   
    ax.scatter(xp-mean[0], yp-mean[1], zp-mean[2], color=c, label=label)


    
    print('radial deviation:',np.linalg.norm(np.array(mean)-np.array([xp,yp,zp])),'mm')

    


u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
r = 0.65  # kriterium MTRAs für AQA
x_circ = r*np.cos(u)*np.sin(v)
y_circ = r*np.sin(u)*np.sin(v)
z_circ = r*np.cos(v)
# alpha controls opacity
ax.plot_surface(x_circ, y_circ, z_circ, color="g", alpha=0.1)


ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.legend()
plt.tight_layout()
plt.show()
