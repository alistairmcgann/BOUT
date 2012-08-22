import numpy as np
import matplotlib.pyplot as plt
from boutdata import collect
from boututils import file_import
from scipy import interpolate

# Load in the data from the file
#path = "/hwdisks/data/bd512/ara/data"

#Apar = collect('Apar', path=path)
#B0 = collect('Bxy', path=path)
#Bt = collect('Btxy', path=path)
Apar = np.load('/hwdisks/data/bd512/ara/apar.npy')[1,:,:,:]
#Apar = np.load('apar.npy')[1,:,:,:]
#Apar *= 5000

#Apar = np.ndarray([256,64,64], dtype=np.float32)
#for i in range(Apar.shape[0]):
#    for k in range(Apar.shape[2]):
#        Apar[i,:,k] = np.exp(-(float(i)/Apar.shape[0] - 0.5)**2 - (float(k)/Apar.shape[2] - 0.5)**2)

#Apar *= 4.

grid_file = file_import('/hwdisks/data/bd512/ara/slab_q0_x_256x64med.nc')

#B0 = collect("B0", path=path)
#Apar = data[0,:,:,:]

#B = np.ndarray(np.shape(Apar))

Bxy  = grid_file['Bxy']
Bpxy = grid_file['Bpxy']
Btxy = grid_file['Btxy']
hthe = grid_file['hthe']
Rxy  = grid_file['Rxy']

dx_xy = grid_file['dx']
dy_xy = grid_file['dy']
dz    = 2.*np.pi/64.

nx = grid_file['nx']
ny = grid_file['ny']

nz = Apar.shape[2]

# Calculate the metric tensor at location x,y
def metric(x,y):
    print x,y
    nu = (Btxy[x,y] * hthe[x,y]) / (Bpxy[x,y]) * (Rxy[x,y])

    g11 = ( Rxy[x,y] * Bpxy[x,y] )**2
    g12 = 0
    g13 = 0 # -1.* I * ( g['Rxy'][x,y] * g['Bpxy'][x,y] )**2
    g22 = 1. / (hthe[x,y]**2)
    g23 = nu / (hthe[x,y]**2)
    g33 = (Bxy[x,y])**2 / (( Rxy[x,y] * Bpxy[x,y])**2 )
    return  np.array([[g11,g12,g13], [g12,g22,g23], [g13,g23,g33]])



#print ap

#def dx_xy(x,y):
#    gridpts = grid_file['dx']
#    intrp = interpolate.RectBivariateSpline(range(gridpts.shape[0]),range(gridpts.shape[1]),gridpts,kx=3,ky=3)
#    tx,ty = intrp.get_knots()
#    tck = (tx,ty,intrp.get_coeffs(),3,3)
#    return interpolate.bisplev(x,y,tck)

#def dy_xy(x,y):
#    gridpts = grid_file['dy']
#    intrp = interpolate.RectBivariateSpline(range(gridpts.shape[0]),range(gridpts.shape[1]),gridpts,kx=3,ky=3)
#    tx,ty = intrp.get_knots()
#    tck = (tx,ty,intrp.get_coeffs(),3,3)
#    return interpolate.bisplev(x,y,tck)

def apar_intrp(x,y,z):
    # Applying a 2D interpolator
#    intrp_consts = interpolate.bisplrep(range(Apar.shape[0]),range(Apar.shape[1]),Apar[:,:,z])
#    intrp_consts = #interpolate.bisplrep(ind[0],ind[1],Apar[:,:,z].flatten()) #interpolate.bisplrep(a_x,a_y,ap)
#    print intrp_consts[0]

    zp = (z+1) % nz
    zm = (z - 1 + nz) % nz
    dApar_dz = (Apar[x,y,zp] - Apar[x,y,zm]) / (2.*dz)

    # Returns the value derivatives at [x,y]
    intrp = interpolate.RectBivariateSpline(range(nx),range(ny),Apar[:,:,z],kx=3,ky=3)#.__call__(x,y)
    tx,ty = intrp.get_knots()
    tck = (tx,ty,intrp.get_coeffs(),3,3)

    dervs = (interpolate.bisplev(x,y,tck, dx=1, dy=0),interpolate.bisplev(x,y,tck, dx=0, dy=1), dApar_dz)

#    print dApar_dz

    return  dervs#interpolate.bisplev(x,y,tck, dx=2, dy=2), dApar_dz#, interpolate.bisplev(x,z,interp_consts,dx=3,dy=3)


def apar_intrp2(x,y,z):
    # Applying a 2D interpolator                                                                                                                                                    
#    intrp_consts = interpolate.bisplrep(range(Apar.shape[0]),range(Apar.shape[1]),Apar[:,:,z])                                                                                     
#    intrp_consts = #interpolate.bisplrep(ind[0],ind[1],Apar[:,:,z].flatten()) #interpolate.bisplrep(a_x,a_y,ap)                                                                    
#    print intrp_consts[0]                                                                                                                                                         

    dx = dx_xy[x,y]
    dy = dy_xy[x,y]
 

    dApar_dy = (Apar[x,y+1,z] - Apar[x,y-1,z]) / (2.*dy)

    # Returns the value derivatives at [x,y]                                                                                                                                        
    intrp = interpolate.RectBivariateSpline(range(nx),range(nz),Apar[:,y,:],kx=3,ky=3)#.__call__(x,y)                                                                               
    tx,ty = intrp.get_knots()
    tck = (tx,ty,intrp.get_coeffs(),3,3)

#    print x, y, z

    dervs = (interpolate.bisplev(x,z,tck, dx=1, dy=0)/dx,dApar_dy, interpolate.bisplev(x,z,tck, dx=0, dy=1)/dz)

#    print dApar_dz                                                                                                                                                                 

    return  dervs#interpolate.bisplev(x,y,tck, dx=2, dy=2), dApar_dz#, interpolate.bisplev(x,z,interp_consts,dx=3,dy=3) 


def f(x,y,z):
# Note that the distances between the points are taken from the grid file, then simply converted to integer
# This really ought to be interpolated.
    dx = dx_xy[x,y]
    dy = dy_xy[x,y]

#    print dx,dy,dz

# Numerical derivatives of A_parallel

    zp = (z+1) % nz
    zm = (z - 1 + nz) % nz
    
    dApar_dx = (Apar[x+1,y,z] - Apar[x-1,y,z]) / (2.*dx)
    dApar_dy = (Apar[x,y+1,z] - Apar[x,y-1,z]) / (2.*dy)
    dApar_dz = (Apar[x,y,zp] - Apar[x,y,zm]) / (2.*dz)

    return [dApar_dx, dApar_dy, dApar_dz]

# Function calculates the direction of the pertubation using
# the derivatives, and then finding the corresponding location
# on the next slice down the tokamak
def follow_field(x,y,z):
    g = metric(x,y)
    dA = apar_intrp2(x,y,z) #f(x,y,z)

#    print 'interp:', dA
#    print 'grid  :', f(x,y,z)

    f1 = ( g[0,2]*dA[0] + g[1,2]*dA[1] + g[2,2]*dA[2] )*(-1./Bxy[x,y])
    f3 = ( g[0,0]*dA[0] + g[1,0]*dA[1] + g[2,0]*dA[2] )*(1./Bxy[x,y])
    
    dxdy = ( g[0,0]*f1 + g[0,2]*f3 ) * (hthe[x,y] / Bpxy[x,y])
    dzdy = ( g[2,0]*f1 + g[2,2]*f3 ) * (hthe[x,y] / Bpxy[x,y])

    return dxdy,dzdy


def fieldline(y, xz):
    x  = xz[0]
    z = (xz[1] + nz) % nz
    dxdy, dzdy = follow_field(x, y, z)
#    print x, y,z, " -> ", dxdy, dzdy
    return [dxdy * dy_xy[x,y]  / dx_xy[x,y] , dzdy * dy_xy[x,y]  / dz]

# Take two random x- and z- co-ordinates to follow the field from.

#x = np.random.random_integers(0, np.shape(Apar)[0]-4)
#z = np.random.random_integers(0, np.shape(Apar)[2]-4)

x = 150
z = 30

#q = (range(Apar.shape[1]-1))
#r = np.ndarray([Apar.shape[1]-1,2])
d = np.ndarray([2,ny-1],dtype=float)
#for y in range(ny-1):
#    d[:,y] = x,z
#    dx_dy, dz_dy = follow_field(x,y,z)
#    x = x + dx_dy * dy_xy[x,y]  / dx_xy[x,y]
#    z = z + dz_dy * dy_xy[x,y]  / dz
#    z = z % nz
#    print "=> ", x, y, z

from scipy.integrate import ode
r = ode(fieldline).set_integrator('vode', first_step = 0.5)
r.set_initial_value([x,z], 0)
d[:,0] = x,z
#for y in range(1,ny-1):
while r.successful() and r.t < nz:
    print r.y, r.t
    r.integrate(r.t+1)
    x = r.y[0]
    z = r.y[1]
    d[:,r.t] = x,z
#    print "=> ", x, y, z

plt.contour(Apar[:,1,:])
plt.contour(interpolate.RectBivariateSpline(range(nx),range(ny),Apar[:,1,:],kx=3,ky=3).__call__(range(nx),range(ny)))

#plt.contour(Apar[:,0,:])
#plt.scatter(30,150)
plt.scatter(d[1],d[0])
#plt.scatter(r[:,0], r[:,1])
#plt.show()
plt.show()
#np.save('/home/adm518/Documents/test.npy', B)
#print B
