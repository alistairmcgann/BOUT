import numpy as np
from mayavi import mlab
from boututils import file_import

#Read in the data file, extracting the variable required
datafile = file_import("data/BOUT.dmp.0.nc")
T = datafile.read("T")
datafile.close()

#Returns the dataset from the variable to be plotted, here for a given time
def T_time(t):
    return T[t,:,:,0]

#Plots the dataset for a given timestep
#This timestep should be confined to the shape of the t dimension of T
t=0
plot = mlab.imshow(T_time(t))
#Eventually, this will update the plot

plot.mlab_source.set(t=tnew)
