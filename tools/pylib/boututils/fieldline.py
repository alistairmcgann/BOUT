import numpy as np
import matplotlib.pyplot as plt
from boutdata import collect
from boututils import file_import

# Load in the data from the file
#path = "/hwdisks/data/bd512/ara/data"

#Apar = collect('Apar', path=path)
#B0 = collect('Bxy', path=path)
#Bt = collect('Btxy', path=path)
#Apar = np.load('/hwdisks/data/bd512/ara/apar.npy')[1,:,:,:]
Apar = np.ndarray([256,64,64], dtype=np.float32)
for i in range(Apar.shape[0]):
    for k in range(Apar.shape[2]):
        Apar[i,:,k] = np.exp(-2*( np.linspace(-2.5,2.5,Apar.shape[0])[i]**2 + np.linspace(-1,1,Apar.shape[2])[k]**2))

grid_file = file_import('/hwdisks/data/bd512/ara/slab_q0_x_256x64med.nc')

#B0 = collect("B0", path=path)
#Apar = data[0,:,:,:]

#B = np.ndarray(np.shape(Apar))

Bxy = grid_file['Bxy']
Bpxy = grid_file['Bpxy']
Btxy = grid_file['Btxy']
hthe = grid_file['hthe']
Rxy = grid_file['Rxy']

def g(x,y):
    nu = (Btxy[x,y] * hthe[x,y]) / (Bpxy[x,y]) * (Rxy[x,y])

    g11 = ( Rxy[x,y] * Bpxy[x,y] )**2
    g12 = 0
    g13 = 0 # -1.* I * ( g['Rxy'][x,y] * g['Bpxy'][x,y] )**2
    g22 = 1. / (hthe[x,y]**2)
    g23 = nu / (hthe[x,y]**2)
    g33 = (Bxy[x,y])**2 / (( Rxy[x,y] * Bpxy[x,y])**2 )
    return  [g11,g12,g13, g12,g22,g23, g13,g23,g33]

def f(x,y,z):
# Note that the distances between the points are taken from the grid file, then simply converted to integer
# This really ought to be interpolated.
    dx = grid_file['dx'][x,y]
    dy = grid_file['dy'][x,y]
    dz = 2.*np.pi/64. #int(grid_file['dz'][x,y])

#    print dx,dy,dz

# Numerical derivatives of A_parallel
#    
#    dApar_dy = (-Apar[x,y-3,z]/60. + 3.*Apar[x,y-2,z]/20. - 3.*Apar[x,y,z]/4 + 3.*Apar[x,y,z]/4 - 3.*Apar[x,y+2,z]/20. + Apar[x,y+3,z]/60.) / 2.#* dy
#    dApar_dz = (-Apar[x,y,z-3]/60. + 3.*Apar[x,y,z-2]/20. - 3.*Apar[x,y,z]/4 + 3.*Apar[x,y,z]/4 - 3.*Apar[x,y,z+2]/20. + Apar[x,y,z+3]/60.) / 2.#* dz

#    print Apar[x+1,y,z] - Apar[x-1,y,z], dx

    dApar_dx = (Apar[x+1,y,z] - Apar[x-1,y,z]) / 2.*dx
    dApar_dy = (Apar[x,y+1,z] - Apar[x,y-1,z]) / 2.*dy
#    print x,y,z
    dApar_dz = (Apar[x,y,z+1] - Apar[x,y,z-1]) / 2.*dz

    return [dApar_dx, dApar_dy, dApar_dz]

def f1(x,y,z):
    u = f(x,y,z)
    v = g(x,y)
    return ( v[2]*u[0] + v[5]*u[1] + v[8]*u[2])*(-1./Bxy[x,y])


def f3(x,y,z):
    u = f(x,y,z)
    v = g(x,y)
    return ( v[0]*u[0] + v[6]*u[1] + v[6]*u[2])*(-1./Bxy[x,y])


# Function calculates the direction of the pertubation using
# the derivatives, and then finding the corresponding location
# on the next slice down the tokamak
def follow_field(x,y,z):
    v = g(x,y)
    dxdy = (v[0]*f1(x,y,z) + v[2]*f3(x,y,z)) * (hthe[x,y] / Bpxy[x,y])
    dzdy = (v[6]*f1(x,y,z) + v[8]*f3(x,y,z)) * (hthe[x,y] / Bpxy[x,y])

    return dxdy,dzdy


# Take two random x- and z- co-ordinates to follow the field from.

#x = np.random.random_integers(0, np.shape(Apar)[0]-4)
#z = np.random.random_integers(0, np.shape(Apar)[2]-4)

x = 150
z = 30

#q = (range(Apar.shape[1]-1))
#r = np.ndarray([Apar.shape[1]-1,2])
d = np.ndarray([2,Apar.shape[1]-1],dtype=int)
for y in range(np.shape(Apar)[1]-1):

#    print x,y,z
    d[:,y] = x,z
#    q[y] = [x,z]

    dx_dy = follow_field(x,y,z)[0]
    dz_dy = follow_field(x,y,z)[1]

#    r[y,0],r[y,1] = dx_dy,dz_dy

    x = x + grid_file['dx'][x,y]*dx_dy #(2.*np.pi/64.)*dx_dy
    z = z + (2.*np.pi/64.)*dz_dy





#for i in range(2,Apar.shape[1]-1):
#    d[:,i] = q[i]#int(q[i][0]),int(q[i][1])

plt.contour(Apar[:,0,:])
#plt.scatter(30,150)
plt.scatter(d[1],d[0])
#plt.scatter(r[:,0], r[:,1])
plt.show()

#np.save('/home/adm518/Documents/test.npy', B)
#print B
