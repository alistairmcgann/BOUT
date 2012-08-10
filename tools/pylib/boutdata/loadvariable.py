import numpy as np
from boututils import DataFile
from boutdata import collect

# Load in the datafile as a BOUT DataFile object
# path links to the directory containing the data
# file1 is the first file in this subdirectory (NB requires / at start)
path = '/hwdisks/home/nrw504/BOUT/blobsims/3DMASTmodel/RUNS/RUN13'
file1 = '/BOUT.dmp.0.nc'

f = DataFile()
print 'Opening file: ' + path + file1
DataFile.open(f, path + file1)

# Calculates the ranges of time, x, y and z in the dataset
# See BOUT manual p.27
t = [int(np.min(f.read('t_array'))), int(np.max(f.read('t_array')))]
nx = f.read('NXPE') * f.read('MXSUB') + 2*f.read('MXG')
ny = f.read('NYPE') * f.read('MYSUB') + 2*f.read('MYG')
nz = f.read('MZ')

# Function query_yes_no released under MIT license
# http://code.activestate.com/recipes/577058-query-yesno/ Accessed 6/8/2012

# Returns 'yes' if user responds 'y', 'no' if 'n', &c.
def query_yes_no(question, default=None):
    """Ask a yes/no question via raw_input() and return their answer.
    
    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is one of "yes" or "no".
    """
    valid = {"yes":"yes",   "y":"yes",  "ye":"yes",
             "no":"no",     "n":"no"}
    if default == None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while 1:
        print question + prompt
        choice = raw_input().lower()
        if default is not None and choice == '':
            return default
        elif choice in valid.keys():
            return valid[choice]
        else:
            print ("Please respond with 'yes' or 'no' "\
                             "(or 'y' or 'n').\n")

# Loop means more than one variable can be loaded
# Perhaps better implemented as a function
c = 0
while c == 0:
    # An iterator to loop over all of the variables in the file
    it = f.handle.variables.__iter__()
    
    print 'Variable Number' + '\t' + 'Name' + '\t' + '\t' + 'Dims.'
    # For all variables in the file, print the identifier in
    # the dictionary, its name and the number of dimensions
    for i in range(np.shape(f.handle.variables.keys())[0]):
        print str(i) + '\t' +  f.handle.variables.keys()[i] + '\t' + '\t' + str(np.shape(f.size(it.next()))[0])
        
    print '\n' + 'Variable number to load?'
    v = raw_input()
    
    print 'Importing variable : ' + f.handle.variables.keys()[v]
    
    # If the variable has 4D data, recommend slicing it up
    if np.shape(f.size(f.handle.variables.keys()[v]))[0] == 4:
        print 'Time slices (-1 for all), Range=', t
        # Takes the Max and Min slice range from the user, places into list
        # that's passed to the collect routine later
        tind = [int(raw_input('min: ')), int(raw_input('max: '))]
        # If the minimum or maximum lie outside the range, just choose everything
        # This is probably not ideal
        if np.min(tind) < 0 or np.max(tind) > t[1]:
            tind = None

        # All of these assume that the minimum spatial point is at 0
        print 'X slices (-1 for all), Range=', [0,nx]
        xind = [int(raw_input('min: ')), int(raw_input('max: '))]
        if np.min(xind) < 0 or np.max(xind) > nx:
            xind = None

        print 'Y slices (-1 for all), Range=', [0,ny]
        yind = [int(raw_input('min: ')), int(raw_input('max: '))]
        if np.min(yind) < 0 or np.max(yind) > ny:
            yind = None

        print 'Z slices (-1 for all), Range=', [0,nz]
        zind = [int(raw_input('min: ')), int(raw_input('max: '))]
        if np.min(zind) < 0 or np.max(zind) > nz:
            zind = None
    
        # Creates a global variable of the same name of the variable in the file
        # The collect routine accumulates all the data using the slices requested
        # Sometimes throws up a memeory error if too much data is requested.
        globals()[str(f.handle.variables.keys()[v])] = collect(f.handle.variables.keys()[v], path=path, tind=tind, xind=xind, yind=yind, zind=zind)
        
    else:
        # If there's less than 4 dimensions, it just grabs the whole lot
        globals()[str(f.handle.variables.keys()[v])] = collect(f.handle.variables.keys()[v], path=path)
        
    # At the users request, a file with the name of the variable and numpy extension can be made
    # in the directory with the data
    if query_yes_no('Save .npy backup?', default='no') == 'yes':
        print 'Saving variable'
        np.save(path + str('/') + str(f.handle.variables.keys()[v].npy), str(f.handle.variables.keys()[v]))

    if query_yes_no('Import another variable?', default='yes') == 'yes':
        c = 0
    else:
        c = 1
