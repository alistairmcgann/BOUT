import numpy as np
from fmlab import two_sfields

data = np.load('/hwdisks/data/adm518/d2.npy')[19,:,:,:]

two_sfields(data,data,spacing = np.array([1., 20., 1.]), no_slices = 9)


