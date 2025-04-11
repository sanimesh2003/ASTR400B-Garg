#In Class Lab 3 Template
# G Besla ASTR 400B

# Load Modules
import numpy as np
import astropy.units as u

# import plotting modules
import matplotlib.pyplot as plt
import matplotlib
%matplotlib inline

# Some Notes about the Isochrone Data
# DATA From   
# http://stellar.dartmouth.edu/models/isolf_new.html
# files have been modified from download.  
# ( M/Mo --> M;   Log L/Lo --> L)
# removed #'s from all lines except column heading
# NOTE SETTINGS USED:  
# Y = 0.245 default   [Fe/H] = -2.0  alpha/Fe = -0.2
# These could all be changed and it would generate 
# a different isochrone


# Filename for data with Isochrone fit for 1 Gyr
# These files are located in the folder IsochroneData
filename1="./IsochroneData/Isochrone1.txt"

# major peak
filename11 = './IsochroneData/Isochrone11.txt'
filename10 = './IsochroneData/Isochrone10.txt'

data11 = np.genfromtxt(filename11,dtype=None,
                      names=True,skip_header=8)

data10 = np.genfromtxt(filename10,dtype=None,
                      names=True,skip_header=8)
# next peak
filename6 = './IsochroneData/Isochrone6.txt'
filename7 = './IsochroneData/Isochrone7.txt'

data6 = np.genfromtxt(filename6,dtype=None,
                      names=True,skip_header=8)

data7 = np.genfromtxt(filename7, dtype=None,
                      names=True,skip_header=8)
# READ IN DATA
# "dtype=None" means line is split using white spaces
# "skip_header=8"  skipping the first 8 lines 
# the flag "names=True" creates arrays to store the date
#       with the column headers given in line 8 

# Read in data for an isochrone corresponding to 1 Gyr
data1 = np.genfromtxt(filename1,dtype=None,
                      names=True,skip_header=8)

# Plot Isochrones 
# For Carina

fig = plt.figure(figsize=(10,10))
ax = plt.subplot(111)

# Plot Isochrones

# Isochrone for 1 Gyr
# Plotting Color vs. Difference in Color 
plt.plot(data1['B']-data1['R'], data1['R'], color='blue', 
         linewidth=5, label='1 Gyr')
###EDIT Here, following the same format as the line above 
plt.plot(data10['B']-data10['R'], data10['R'], color='red', 
         linewidth=5, label='10 Gyr')
plt.plot(data11['B']-data11['R'], data11['R'], color='yellow', 
         linewidth=5, label='11 Gyr')
plt.plot(data6['B']-data6['R'], data6['R'], color='magenta', 
         linewidth=5, label='6 Gyr')
plt.plot(data7['B']-data7['R'], data7['R'], color='green', 
         linewidth=5, label='7 Gyr')

# Add axis labels
plt.xlabel('B-R', fontsize=22)
plt.ylabel('M$_R$', fontsize=22)

#set axis limits
plt.xlim(-0.5,2)
plt.ylim(5,-2.5)

#adjust tick label font size
label_size = 22
matplotlib.rcParams['xtick.labelsize'] = label_size 
matplotlib.rcParams['ytick.labelsize'] = label_size

# add a legend with some customizations.
legend = ax.legend(loc='upper left',fontsize='x-large')

#add figure text
plt.figtext(0.5, 0.15, 'CMD for Carina dSph', fontsize=22)

plt.savefig('IsochroneCarina.png')


import os

html_file = "Lab3 Garg.html"
ipynb_file = "Lab3 Garg.ipynb"

# Delete old HTML if it exists
if os.path.exists(html_file):
    os.remove(html_file)

# Convert notebook to HTML
!jupyter nbconvert --to html "{ipynb_file}"
