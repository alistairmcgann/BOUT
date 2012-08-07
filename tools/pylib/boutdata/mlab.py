import numpy as np
from mayavi.api import Engine
import mayavi.mlab as mlab
#from boutdata import collect


# Load the data in from BOUT output files.

#path = "/hwdisks/data/adm518/data_33/"

#data = collect("P", tind=50, path=path)
#data = data[0,:,:,:]
#s = np.shape(data)
#nz = s[2]

# Save a backup of the data to be plotted
# Avoids a massive reload if something goes wrong
#np.save('data.npy', data)

data = np.load('/hwdisks/data/adm518/density.npy')[0,:,:,:]

# Creating the mlab scene which will contain all of the visualisations
plot = mlab.figure()

# Loads in the data as a scalar field, creates the VTK data source
# Can be changed for other types of data source (vector field, etc)
field = mlab.pipeline.scalar_field(data)

# Changes the spacing of the grid points to make the data look right
# fiddle with until it looks good.
field.spacing = np.array([1., 20., 1.])


#Creates the 'cloud' of the whole data.
mlab.pipeline.volume(field, figure=plot)

# Adds in the 'slices' within the cloud.
# ALWAYS use image_plane_widget over scalar cut plane, much quicker (ref. Docs)
no_slices = 9

for n in np.around(range(no_slices)):
    mlab.pipeline.image_plane_widget(field, figure=plot, plane_orientation = 'y_axes', slice_index=( np.int((n + 3./4.)*((np.shape(data)[2]/no_slices)) ) ) )

mlab.draw(figure=plot)

#mlab.savefig('./fig.png', figure=plot, magnification=2)
