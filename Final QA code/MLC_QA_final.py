# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 11:27:54 2023

@author: SchockSinjaTheresa
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks_cwt
from scipy.ndimage import gaussian_filter, median_filter
import os
#import tifffile as tf
from helper_final import (file_einlesen, get_nearest, interpolate, circle_finder)

####################
#  Daten Einlesen  #
####################
path = 'Data examples/MLC QA example/Integral 24_07_2023 14_52_04.88.opg'  #path to file including filename
SAD = 796.8  #measured SAD

data, x, y, _ = file_einlesen(path)

plt.close()
plt.imshow(data)
plt.show()


max_data = np.max(data)
width=0.4*800/(SAD)
x = x/0.4*width
y = y/0.4*width

###################
#  Marker suchen  #
###################
circles_ind, circles = circle_finder(data,x,y, width)
lowercirc = circles[:,0]
mid = circles[:,1]

# optional flatfield correction for Picket Fence?
#data = data/flatfield
#data = data/np.max(data)

#######################
#  Berechne Rotation  #
#######################
def linear(x, a, b):
    return a*x+b


pop, pcov = curve_fit(linear, circles[1,:], circles[0,:])
print('Winkel=', np.rad2deg(np.arctan(pop[0])),'deg')
alpha = np.arctan(pop[0]) #in radiant
print('middle:', mid)


#########################
#  Erstelle Leave-Grid  #
#########################
#Soll: centered, dann rotiert (math. positiv korrespondiert zu Winkeldefinition oben), dann geshifted (x' = xcos(a)+ysin(a)+x_0,   y' = -xsin(a)+ycos(a)+y_0)
x_leaves = np.array([-55,-45,-30,-20,-5,5,20,30,45,55])
y_leaves = np.linspace(-12.5,12.5, 26)*3.85   #26 leaves, Breite 3.85mm bei 800mmSAD
leaves = np.ones((26,10,2))

for i in range(26):
    x_prime = x_leaves*np.cos(alpha) + y_leaves[i]*np.sin(alpha) + mid[0]
    y_prime = -x_leaves*np.sin(alpha) + y_leaves[i]*np.cos(alpha) + mid[1]
    leaves[i,:,0] = x_prime
    leaves[i,:,1] = y_prime

###################################################
#  Evaluiere Leafpositionen auf rotierten Achsen  #
###################################################
left_leaves = np.zeros((26, 5))
right_leaves = np.zeros((26, 5))
left_leaves_bayouth = np.zeros((26, 5))
right_leaves_bayouth = np.zeros((26, 5))
j = 0
for value in y_leaves:

    prof_y_ind = (-x*np.sin(alpha)+value*np.cos(alpha)+mid[1])/width +len(y)/2
    prof_y_ind = prof_y_ind.astype(int)
    leaf_profile = data[np.arange(len(x)), prof_y_ind]


    peaks2 = find_peaks_cwt(leaf_profile, widths=20.0)  #Filter Daten um Peaks zu finden
    if len(peaks2) != 5:
        print('%d peaks detected, y=%i' % (len(peaks2), value))  #Fehler ist raised, wenn der Filter zu stark oder zu schwach ist

    Halfmax = leaf_profile[peaks2]/2
    l = np.array([])
    r = np.array([])
    k=0
    for val in peaks2:
        l_border = 30
        r_border=30
        if val<30:
            l_border=val
        elif (len(leaf_profile)-val)<30:
            r_border = len(leaf_profile)-val

        test_profile = leaf_profile[val-l_border:val+r_border]  #Teilprofil, auf dem nach leafposition gesucht wird
        test_x = x[val-l_border:val+r_border]
        [left, right] = interpolate(test_profile, test_x, (leaf_profile[val])*0.5)
        l = np.append(l, left)
        r = np.append(r, right)
        k+=1
    left_leaves[j, :] = l
    right_leaves[j, :] = r
    j += 1
    #print('stripes width:',(r-l)/np.cos(alpha)) #Kann bei Interesse angezeigt werden
    #print('gap width:', (l[1:]-r[:-1])/np.cos(alpha))
    

##############
#  QA Check  #
##############
diff_l = leaves[:,:-1:2,0] - left_leaves
diff_r = leaves[:,1::2,0] - right_leaves

Fail_1_l = np.argwhere(np.abs(diff_l)>0.5)
Fail_1_r = np.argwhere(np.abs(diff_r)>0.5)
Fail_2_l = np.argwhere(np.abs(diff_l)>0.95)
Fail_2_r = np.argwhere(np.abs(diff_r)>0.95)
Check1l = len(Fail_1_l) 
Check1r = len(Fail_1_r)
Check2 = len(Fail_2_l)+ len(Fail_2_r)
max_diff = np.max(np.abs(np.append(diff_l, diff_r)))


Fail_l = np.argwhere(np.abs(diff_l.flatten())>0.5)
Fail_r = np.argwhere(np.abs(diff_r.flatten())>0.5)

if np.all(np.array([Check1l, Check1r]))<=13 and Check2 == 0:
    print('MLC QA passed. %d leaves with difference>0.5mm in X1, %d leaves >0.5mm in X2, 0 leaves with difference>0.95mm. Maximal Difference: %f mm' %(Check1l, Check1r, max_diff))
else:
    print('MLC QA failed. %d leaves with difference>0.5mm in X1, %d leaves >0.5mm in X2, %d leaves>0.95mm' %(Check1r, Check1l, Check2))
if Check1l+Check1r != 0:
    print('Failing leaves on right bank (X1):', diff_r.flatten()[Fail_r])
    print('Failing leaves on left bank (X2):', diff_l.flatten()[Fail_l])
    
 #%%
    
###################

#  plot like RIT  #
###################
y_pos = np.arange(26)+1
x_pos = np.linspace(0,4,17)*3


#right (X1)
plt.close()
for i in range(5):
    leaves = diff_r[:,i]
    fail = np.argwhere(np.abs(leaves)>0.5)
    color = ['green']*26
    for idx in fail:
        color[int(idx)]='red'
    plt.barh(y_pos, leaves, color = color, left = i*2+1) 
x_pos = np.linspace(0,10,21)
plt.xticks(x_pos, [-1,-0.5,0,0.5,+-1,-0.5,0,+0.5,1,-0.5,0,+0.5,1,-0.5,0,+0.5,1,-0.5,0,+0.5,+1], weight='bold')
plt.yticks(y_pos[::2], y_pos[::-2], weight='bold')  #auch Dartsellungssache. Ich habe nicht die Daten gespiegelt, sondern nur die Beschriftung. Leaf 26 ist jetzt das Leaf richtung marker
plt.ylabel('leaves', weight='bold')
plt.xlabel('dx in mm', weight='bold')
plt.grid()
plt.ylabel('leaves', weight='bold')
plt.xlabel('dx in mm', weight='bold')
plt.title('Deviations X1 (right bank)', weight='bold')
plt.tight_layout()
plt.show()

plt.close()

#left (X2)
for i in range(5):
    leaves = diff_l[:,i]
    fail = np.argwhere(np.abs(leaves)>0.5)
    color = ['green']*26
    for idx in fail:
        color[int(idx)]='red'
    plt.barh(y_pos, leaves, color = color, left = i*2+1) 
x_pos = np.linspace(0,10,21)
plt.xticks(x_pos, [-1,-0.5,0,0.5,+-1,-0.5,0,+0.5,1,-0.5,0,+0.5,1,-0.5,0,+0.5,1,-0.5,0,+0.5,+1], weight='bold')
plt.yticks(y_pos[::2], y_pos[::-2], weight='bold')  #auch Dartsellungssache. Ich habe nicht die Daten gespiegelt, sondern nur die Beschriftung. Leaf 26 ist jetzt das Leaf richtung marker
plt.ylabel('leaves', weight='bold')
plt.xlabel('dx in mm', weight='bold')
plt.grid()
plt.ylabel('leaves', weight='bold')
plt.xlabel('dx in mm', weight='bold')
plt.title('Deviations X2 (left bank)', weight='bold')
plt.tight_layout()
plt.show()
