import numpy as np
from boutdata import collect
from boututils import file_import

# Load in the data from the file
#path = "/hwdisks/data/bd512/ara/data"

#Apar = collect('Apar', path=path)
#B0 = collect('Bxy', path=path)
#Bt = collect('Btxy', path=path)
Apar = np.load('/hwdisks/data/bd512/ara/apar.npy')
grid_file = file_import('/hwdisks/data/bd512/ara/slab_q0_x_256x64med.nc')

#B0 = collect("B0", path=path)
#Apar = data[0,:,:,:]

B = np.ndarray(np.shape(Apar))

def g(x,y):
    nu = (grid_file['Btxy'][x,y] * (grid_file['hthe'][x,y])) / ((grid_file['Bpxy'][x,y]) * (grid_file['Rxy'][x,y]))

    g11 = ( grid_file['Rxy'][x,y] * grid_file['Bpxy'][x,y] )**2
    g12 = 0
    g13 = 0 # -1.* I * ( g['Rxy'][x,y] * g['Bpxy'][x,y] )**2
    g22 = 1. / (grid_file['hthe'][x,y])**2
    g23 = nu / (grid_file['hthe'][x,y]**2)
    g33 = (grid_file['Bxy'][x,y])**2 / ( (grid_file['Rxy'][x,y] * grid_file['Bpxy'][x,y])**2 )
    return [ [g11,g12,g13], [g12,g22,g23], [g13,g23,g33] ] #g['Rxy[x,y]']

def f(x,y):
# Note that the distances between the points are taken from the grid file, then simply converted to integer
# This really ought to be interpolated.
    dx = int(grid_file['dx'][x,y])
    dy = int(grid_file['dy'][x,y])
    dz = 2.*np.pi/64. #int(grid_file['dz'][x,y])

# Numerical derivatives of A_parallel
    dApar_dx = -Apar[x-3*dx,y,z]/60. + 3.*Apar[x-2*dx,y,z]/20. - 3.*Apar[x-dx,y,z]/4 + 3.*Apar[x+dx,y,z]/4 - 3.*Apar[x+2*dx,y,z]/20. + Apar[x+3*dx,y,z]/60.
    dApar_dy = -Apar[x,y-3*dy,z]/60. + 3.*Apar[x,y-2*dy,z]/20. - 3.*Apar[x,y-dy,z]/4 + 3.*Apar[x,y+dy,z]/4 - 3.*Apar[x,y+2*dy,z]/20. + Apar[x,y+3*dy,z]/60.
    dApar_dz = -Apar[x,y,z-3*dz]/60. + 3.*Apar[x,y,z-2*dz]/20. - 3.*Apar[x,y,z-dz]/4 + 3.*Apar[x,y,z+dz]/4 - 3.*Apar[x,y,z+2*dz]/20. + Apar[x,y,z+3*dz]/60.

    return [dApar_dx, dApar_dy, dApar_dz]

def f1(x,y):
    return (-1./grid_file['Bxy'][x,y]) * np.dot([g(x,y)[1,3], g(x,y)[2,3], g(x,y)[3,3]], f(x,y))

def f3(x,y):
    return (1./grid_file['Bxy'][x,y]) * np.dot([g(x,y)[1,1], g(x,y)[2,1], g(x,y)[3,1]], f(x,y))

# Function calculates the direction of the pertubation using
# the derivatives, and then finding the corresponding location
# on the next slice down the tokamak
def follow_field(x,y,z):

    dxdy = g(x,y)[1,1]*f1(x,y) + g(x,y)[1,3]*f3(x,y)
    dzdy = g(x,y)[3,1]*f1(x,y) + g(x,y)[3,3]*f3(x,y)

    B[x+int(dxdy),y,z+int(dzdy)] = B0[x,z]

    x = int(x + dxdy)
    z = int(z + dzdy)

    return x,z

# Take two random x- and z- co-ordinates to follow the field from.

x = np.random.random_integers(0, np.shape(Apar)[0]-4)
z = np.random.random_integers(0, np.shape(Apar)[2]-4)

for y in range(np.shape(Apar)[1]):

    x,z = follow_field(x,y,z)

print B
