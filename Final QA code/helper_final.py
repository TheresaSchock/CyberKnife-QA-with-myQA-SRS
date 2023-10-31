import matplotlib.pyplot as plt
import numpy as np
import codecs
import scipy.interpolate
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import signal
from scipy.ndimage import gaussian_filter1d





def file_einlesen(filename):

    dose_data = np.zeros((300,350))
    x_data = []
    y_data = []
    da=[]

    rawdata = codecs.open(filename, 'r', 'iso-8859-15').readlines() #File zeilenweise einlesen, Liste erstellen
    i=0
    for i in range(len(rawdata)):
        
        #Temperatur auslesen
        if 'Temperature' in rawdata[i]:
            Temperature = (float((rawdata[i][20:])))
            
        #if 'Gantry Angle' in rawdata[i]:
        #    angle = float(rawdata[i][18:])
            
        if '<asciibody>' in rawdata[i]:
            beg=i #Ende des Anfangskörpers der Datei. Danach kommen noch 355 Zeilen mit Daten.

        #Liste mit x-Koordinaten erstellen
        if 'X[mm]' in rawdata[i]:
            j=0
            for j in range(0,300,1):
                x_data.append(float((rawdata[i][13+j*5+j*4:13+5+j*4+j*5])))


        if 'Y[mm]' in rawdata[i]:

            y=0
            x=0

            
            while i<beg+355: #letzte Zeile mit Messwerten neu Zeile 405! aber break funktioniert, daher größerer Wert.
                k=0
                i+=1
                if '</asciibody>' in rawdata[i]: 
                    break
                #Liste mit y-Koordinaten erstellen
                y_data.append(float(rawdata[i][4:9]))

                j=0
                
                #for j in range(0,2702,1): #Spalten mit Messwerten
                for j in range(0,4000,1): #es müssen pro Zeile 300 Messwerte sein

                    if '\t' in rawdata[i][j]:
                        k+=1
                        candidate = str(rawdata[i][j+1:j+11]) #theoretisch werte mit 4 Nachkommastellen, aber keine Nullen ausgeschrieben, darum sind manche Einträge 9, manche 11 stellen lang
                        new = candidate.split('\t')
                        #dose_data[x,y] = float(rawdata[i][j+1:j+9]) 
                        dose_data[x,y] = float(new[0])
                        
                        x+=1
                    j+=1
                    if k==300:
                        break

                x=0
                y+=1
   

        if '</asciibody>' in rawdata[i]: 
            break

    return(np.array(dose_data, dtype=np.float64), np.array(x_data), np.array(y_data), Temperature)
    

def get_nearest(data, border,n):
    #n: number of nearest points
    dist = np.abs(data-border)
    ind = np.sort(list(np.argsort(dist)[:n]))
    #left do linear interpolation
    points = data[ind]
    return points, ind

def get_nearest_border(data,border,n):
    min_ind = np.argmin(np.abs(data-border))
    ind_test= np.linspace(min_ind-n/2, min_ind+n/2, n+1, dtype=int)
    if data[min_ind]<border and data[min_ind-1]<data[min_ind]:
        ind=ind_test[1:]
    elif data[min_ind]>border and data[min_ind-1]>data[min_ind]:
        ind=ind_test[1:]
    else:
        ind=ind_test[:-1]
    #print(len(data), ind) ###weg
    points = data[ind]
    return points, ind


    

def interpolate (data, coords, border, integrate = False, int_test = False, c=0):
    #left
    results = []
    l = int(np.argmax(data))
    y_interp_left = scipy.interpolate.interp1d(data[:l], coords[:l])

    #right
    y_interp_right = scipy.interpolate.interp1d(data[l:], coords[l:])
    left =  y_interp_left(border)
    right = y_interp_right(border)
    results.extend([left,right])
    if int_test:  #nur für interpolationstest
    ###hier get_nearest_border###
        left_values, ind_l = get_nearest_border(data[:l], border,4)
        right_values, ind_r = get_nearest_border(data[l:], border,4)
        results.extend([left_values, right_values])
        results.extend([ind_l,ind_r+l])
    if integrate:
        finterp = InterpolatedUnivariateSpline(coords, data, k=1)
        a = finterp.integral(left, c) #c=0
        b = finterp.integral(c,right) #c=0
        results.extend([a, b])
    return results

def Flatness_siemens(profile, FW, center, coords, plot=False):
    interp = scipy.interpolate.interp1d(coords, profile) #alternativ fill_value='extrapolate'
    center=center*0.4+coords[0]
    x = np.linspace((center-0.4*FW), (center+0.4*FW), 50) #flattened field defined as 80% of field width
    flat_data = interp(x)
    if plot:
        plt.close()
        plt.plot(coords, profile)
        plt.plot(x, flat_data)
        plt.show()
    D_max = np.max(flat_data)
    D_min = np.min(flat_data)
    flatness = np.abs(D_max-D_min)*100/(D_max+D_min) #IBA Siemens
    if D_min !=0: 
        flat_PTW = D_max*100/D_min #PTW
    else: flat_PTW = 0
    return np.round(flatness, decimals = 2), np.round(flat_PTW, decimals = 2)

def symmetry(a,b):
    sym = np.abs(a-b)*100/np.abs(a+b)
    return sym       

def get_profiles(data, x, y, cax_correction = False):
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
    if cax_correction:
        [left, right] = interpolate(profile_x, x, half_dose)
        Field_center_x = (left+right)/2
        c_x_val, c_x = get_nearest(x,Field_center_x,1)
        profile_y = np.array(data[c_x, :].reshape(350))
        
        [left, right] = interpolate(profile_y, y, half_dose)
        Field_center_y = (left+right)/2
        c_y_val, c_y = get_nearest(y,Field_center_y,1)
        profile_x = np.array(data[:,c_y].reshape(300))
        half_dose = data[c_x, c_y].squeeze(0)*0.5
    return [profile_x, profile_y], [c_x_val,c_y_val],[c_x,c_y], half_dose


def circle_finder(data,x,y, width, analysis=False, x_int=[],y_int=[], m=0):
    #widht = pixelgröße in mm: 0.4mm
    d_max = np.max(data)
    Start_found = False
    End_Found = False
    for i in range(len(y)):
        if not Start_found:
            if np.max(data[:, i]) > 0.1*d_max: #eig 0.1dmax
                Start_y = i  # bei 0.1 direkt werden nicht alle peaks entdeckt. besser mit gefilterten Daten??-> nur wenn height zu hoch eingestellt ist bei peakfinder!!
                Start_found = True
        else:
            if i>Start_y+100 and np.max(data[:, i]) < 0.1*d_max:
                End_y = i
                End_Found = True
                break
            
    if not End_Found:
        End_y = len(y)
    cropped_data = data[:, Start_y:End_y]


    #search for rows with marker by detecting 6 maxima in gauss-filtered data
    marker_ind = [[], []]
    min_pos = [[], []]
    max_pos = [[], []]
    marker_found = False #Flag to distinguish between the two markers
    i = -1
    
    for k in range(cropped_data.shape[1]):
        dataFiltered = gaussian_filter1d(
            cropped_data[:, k].astype('float64'), sigma=3)
        tMax = signal.argrelmax(dataFiltered)[0]
        tMin = signal.argrelmin(dataFiltered)[0]

        #If proper minimum is detected (not due to noise)
        if len(tMax) == 6 and np.all(cropped_data[tMax[2:4], k] > (np.ones(2)*(cropped_data[tMin[2], k]+30))):
            if not marker_found:
                i += 1
            marker_ind[i].append(k+Start_y)
            min_pos[i].append(tMin[2])
            max_pos[i].append(tMax[2:4])
            marker_found = True
        else:
            marker_found = False

    if i==-1:
        print('no markers found')
        return [],[]
    # y_value: search for FWHM in central row

    y_val = np.zeros(2)

    for n in [0, 1]:
        i = int(np.mean(min_pos[n]))

        search_data = (data[i, marker_ind[n]]-data[i, marker_ind[n][0]])*(-1) #cut and transform data to part with dip
        #search_data = search_data/np.max(search_data)
        border = np.max(search_data)*0.5
        if analysis and n==1:
            [l, r, _, _,ind_l,ind_r] = interpolate(search_data, y[marker_ind[n]], border, int_test=True)
            y_int[m,:10]=search_data[:10]
        else:
            [l, r] = interpolate(search_data, y[marker_ind[n]], border)
        y_val[n] = ((r-l)/2+l) #absolute position
        d_50=border
        #print('y:',(y_val[n]+69.8)/0.4) #in pixel


    # x_value: equivalent
    x_val = np.zeros(2)
    for n in [0, 1]:
        p,i = get_nearest(y, y_val[n],1)
        search_data = (data[max_pos[n][0][0]:max_pos[n][0][1]+1, i]-data[max_pos[n][0][0], i])*(-1)
        #search_data = search_data/np.max(search_data)
        search_data = search_data.flatten()
        border = np.max(search_data)*0.5

        if analysis and n==1:

            [l, r, _, _,ind_l,ind_r] = interpolate(search_data, x[max_pos[n][0][0]:max_pos[n][0][1]+1].flatten(), border, int_test=True)
            #x_int[m,:4] = search_data[ind_l]
            #x_int[m,4:] = search_data[ind_r]
            x_int[m, :10] = search_data[:10]
        else:
            [l, r] = interpolate(search_data.flatten(), x[max_pos[n][0][0]:max_pos[n][0][1]+1].flatten(), border)
        x_val[n] = ((r-l)/2+l) #absolut pos
        #d_50 = border
        #print('x:', (x_val[n]+59.8)/0.4) #in pixels
    
  
    x_0 = -x[0]
    y_0=-y[0]
    
    plt.close()
    plt.imshow(data)#, extent=[-69.8, 69.8+1, -59.8, 59.8+1])
    plt.plot((y_val+y_0)/width, (x_val+x_0)/width, marker='.', linewidth=0, color='red')
    plt.show()
    if analysis:
        return np.array([(x_val+x_0)/width,(y_val+y_0)/width]), np.array([x_val, y_val]), x_int, y_int,ind_l, ind_r, d_50
    return np.array([(x_val+x_0)/width,(y_val+y_0)/width]), np.array([x_val, y_val])


        
