import numpy as np
from scipy import interpolate, integrate

def field_trace(data, grid_file, x_0=-1, z_0=-1):

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

    def apar_intrp(x,y,z):

        dx = dx_xy[x,y]
        dy = dy_xy[x,y]
        
        
        # Interpolating down the y-axis (along the field)
        dy_coeffs = interpolate.splrep(range(ny), data[x,:,z], k=3)
        
        # Interpolating along the slices of data
        intrp = interpolate.RectBivariateSpline(range(nx),range(nz),data[:,y,:],kx=3,ky=3)
        
        tx,ty = intrp.get_knots()
        tck = (tx,ty,intrp.get_coeffs(),3,3)
        
        # From the cubic spline coefficients, returns derivatives
        dervs = ( interpolate.bisplev(x,z,tck, dx=1, dy=0)/dx,
                  interpolate.splev(y,dy_coeffs,der=1)/dy,
                  interpolate.bisplev(x,z,tck, dx=0, dy=1)/dz )

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

    r = integrate.ode(fieldline).set_integrator('dopri5', first_step = 0.5)
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
        try:
            d[:,r.t] = x,y,z,data[x,y,z]
        except IndexError:
            d[:,y:] = np.nan
            print x_0,z_0
            break

    return d
