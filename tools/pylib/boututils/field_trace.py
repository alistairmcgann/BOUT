import numpy as np
from scipy import interpolate, integrate, optimize

def field_trace(data, grid_file, x_0=-1, z_0=-1, tck = None, full_output = None):
    ''' Returns a 2D array containing 3
    co-ordinates of a line of constant field value down the y-axis of
    a set of data.

x_0 and z_0 are the initial x and z positions of the field line, the
value at which is held constant throughout the line. If less than
zero, or outside the range of data passed to it, random starting
points are chosen for the data.  The forth dimension of the output is
the value of the data at that (x,y,z) location, which should be
constant.

If interpolation of the data has already been performed, the spline
knots and cubic coefficients can be passed through the 'tck'
argument. If these values are required, because the data does not
change, they can be produced by setting the 'full_output' keyword
argument to 1, which produces a suitable array as its second output.'''

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

#    nx,ny,nz = data.shape
    nz = data.shape[2]

# Calculate the metric tensor at location x,y
    def metric(x,y):
        #    print x,y
        nu = (Btxy[x,y] * hthe[x,y]) / (Bpxy[x,y]) * (Rxy[x,y])
        
        g11 = ( Rxy[x,y] * Bpxy[x,y] )**2
        g12 = 0
        g13 = 0 # -1.* I * ( g['Rxy'][x,y] * g['Bpxy'][x,y] )**2
        g22 = 1. / (hthe[x,y]**2)
        g23 = nu / (hthe[x,y]**2)
        g33 = (Bxy[x,y])**2 / (( Rxy[x,y] * Bpxy[x,y])**2 )
        return  np.array([[g11,g12,g13], [g12,g22,g23], [g13,g23,g33]])

    # Perform all of the interpolation, store the coefficients in 
    # list 'tck', which is called when any values in the [x,z] plane
    # are required

    if tck is None:

#        del tck
        tck = range(ny)
        for j in tck:
            #        print nx, j, data.shape
            intrp = interpolate.RectBivariateSpline(range(nx),range(nz),data[:,j,:],kx=3,ky=3)
            tx,ty= intrp.get_knots()
            np.shape((tx,ty,intrp.get_coeffs(),3,3))
            tck[j] = (tx,ty,intrp.get_coeffs(),3,3)

            #    np.shape(tck[1])

    def apar_intrp(x,y,z):

        dx = dx_xy[x,y]
        dy = dy_xy[x,y]
        
        
        # Interpolating down the y-axis (along the field)
        y_vals = np.array(range(ny), dtype=float)

        for j in range(len(y_vals)):
            y_vals[j] = interpolate.bisplev(x,z,tck[j])
#        print y_vals, 'sdf'
        dy_coeffs = interpolate.splrep(range(ny), y_vals, k=3)
        
        # Interpolating along the slices of data
#        intrp = interpolate.RectBivariateSpline(range(nx),range(nz),data[:,y,:],kx=3,ky=3)
        
#        tx,ty = intrp.get_knots()
#        tck = (tx,ty,intrp.get_coeffs(),3,3)
#        print y, int(y), np.shape(data[:,y,:])
        # From the cubic spline coefficients, returns derivatives
#        print 's', int(np.rint(y)), int(y)
        dervs = ( interpolate.bisplev(x,z,tck[int(np.rint(y))], dx=1, dy=0)/dx,
                  interpolate.splev(y,dy_coeffs,der=1)/dy,
                  interpolate.bisplev(x,z,tck[int(np.rint(y))], dx=0, dy=1)/dz )

        return  dervs

# From the metric tensor and derivatives returns the pertubation
    def follow_field(x,y,z):
        g = metric(x,y)
        dA = apar_intrp(x,y,z)

        f1 = ( g[0,2]*dA[0] + g[1,2]*dA[1] + g[2,2]*dA[2] )*(-1./Bxy[x,y])
        f3 = ( g[0,0]*dA[0] + g[1,0]*dA[1] + g[2,0]*dA[2] )*(1./Bxy[x,y])
    
        dxdy = ( g[0,0]*f1 + g[0,2]*f3 ) * (hthe[x,y] / Bpxy[x,y])
        dzdy = ( g[2,0]*f1 + g[2,2]*f3 ) * (hthe[x,y] / Bpxy[x,y])

        return dxdy,dzdy


    def fieldline(y, xz):
        x  = xz[0]
        z = (xz[1] + nz) % nz
        dxdy, dzdy = follow_field(x, y, z)
        return [dxdy * dy_xy[x,y]  / dx_xy[x,y] , dzdy * dy_xy[x,y]  / dz]

    if x_0 < 0 or x_0 > nx:
        x_0 = np.random.randint(0,nx)
    if z_0 < 0 or z_0 > nz:
        z_0 = np.random.randint(0,nz)

    d = np.ndarray([4,ny-1],dtype=float)

    r = integrate.ode(fieldline).set_integrator('dopri5', first_step = 0.5, max_step=0.5)
    r.set_initial_value([x_0,z_0], 0)
    d[:,0] = x_0,0,z_0,data[x_0,0,z_0]
    for y in range(1,ny-1):
        try:
            r.integrate(y)
        except IndexError:
            d[:,y:] = np.nan
            print x_0,z_0
            break
        x = r.y[0]
        z = r.y[1]

#        if x<data.shape[0] and z<data.shape[2]:
#            print x, z, data.shape[0], data.shape[2]
#            (xd, zd, t) = optimize.fsolve(diff,[x,z],args=(y))
#        else:
#            t = 0

#        if t==1:
#            x = xd
#            z = zd

        try:
            d[:,r.t] = x,y,z,data[x,y,z]
        except IndexError:
            d[:,y:] = np.nan
            print x_0,z_0
            break

    if full_output is not None:
        return d, tck
    else:
        return d
