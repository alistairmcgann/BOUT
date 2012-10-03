import numpy as np
import mayavi.mlab as mlab
def fmlab(volume, slices, spacing=np.array([1.,1.,1.]), no_slices=5, filename=None, size=[1024,768]):
    '''Creates a volume with slices of data from up to 2 scalar fields.

Volume and slices are scalar fields that are used to generate their respective
parts of the plot. They can be the same field, but must be 4- or 3-Dimensional.
If a field is 4-Dimensional, however, only the first point is used. It's best to
pass only 3D data to this.
Spacing is the 'streching factor' of the data in x,y,z format.
no_slices determines the number of slices that are generated. '''
# Only plot one time slice, so remove 4th dimension
    if np.shape(volume) == 4:
        volume = volume[0,:,:,:]

    if np.shape(slices) == 4:
        slices = slices[0,:,:,:]


# Creating the mlab scene which will contain all of the visualisations
    plot = mlab.figure()

# Loads in the data as a scalar field, creates the VTK data source
# Can be changed for other types of data source (vector field, etc)
    s_field1 = mlab.pipeline.scalar_field(volume)
    s_field2 = mlab.pipeline.scalar_field(slices)

# Changes the spacing of the grid points to make the data look right
# fiddle with until it looks good.
    s_field1.spacing = spacing
    s_field2.spacing = spacing

#Creates the 'cloud' of the whole data.
    mlab.pipeline.volume(s_field1, figure=plot)

# Adds in the 'slices' within the volume

    for n in np.around(range(no_slices)):
        mlab.pipeline.image_plane_widget(s_field2, figure=plot, plane_orientation = 'y_axes', slice_index=( np.int((n + 3./4.)*((np.shape(slices)[1]/no_slices)) ) ) )

    if filename is not None:
        mlab.options.offscreen = True
        mlab.draw(figure = plot)
        mlab.savefig(str(filename) + '.png', size=size, figure=plot, magnification=1.)
    else:
        mlab.draw(figure=plot)

        mlab.show()
