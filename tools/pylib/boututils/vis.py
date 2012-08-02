import numpy as np
from collections import deque
from mayavi import mlab
import gc

from mayavi.api import Engine
from traits.api import HasTraits, Range, Instance, on_trait_change
from traitsui.api import View, Item, Group
from mayavi.core.api import PipelineBase
from mayavi.core.ui.api import MayaviScene, SceneEditor, MlabSceneModel

engine = Engine()
engine.start()
engine.new_scene()
mlabscene = engine.scenes[0]
mlab.show_pipeline()

data = np.load('/home/adm518/Documents/dtime.npy')

lr_factor = [1,1,5,1]

print np.shape(data)

print 'lr_factor:' + str(lr_factor[2])

#def lr(t,i,j,k):
#    print t*lr_factor[0]# + i*lr_factor[1] +  j*lr_factor[2] + k*lr_factor[3] 
#    s = np.int(np.asscalar(t*lr_factor[0]))
#    x = np.int(np.asscalar(i*lr_factor[1]))
#    y = np.int(np.asscalar(j*lr_factor[2]))
#    z = np.int(np.asscalar(k*lr_factor[3]))
#    print s,x,y,z
#    return data[s,x,y,z]

low_res = np.ndarray(np.asarray(np.shape(data))[:] / np.asarray(lr_factor)[:], dtype=float)

print np.max(range(np.shape(low_res)[0]))
print np.max(range(np.shape(low_res)[1]))
print np.max(range(np.shape(low_res)[2]))
print np.max(range(np.shape(low_res)[3]))

print (np.shape(low_res))

for t in range(np.shape(low_res)[0]):#a:#range((np.shape(low_res))[0]):
    for i in range(np.shape(low_res)[1]):
        for j in range(np.shape(low_res)[2]):
#            print j
            for k in range( np.shape(low_res)[3] ):

#                print 'b' + str(np.shape(low_res[t,i,j,k]))
#                print 'c' + str(np.shape(data[ t*lr_factor[0],i*lr_factor[1], j*lr_factor[2], k*lr_factor[3] ]))
                low_res[t,i,j,k] = data[ t*lr_factor[0],i*lr_factor[1], j*lr_factor[2], k*lr_factor[3] ]#lr(t,i,j,k)
#print low_res

#low_res = np.ndarray(np.asarray(np.asarray(np.shape(data))[:] / np.asarray(lr_factor)[:] ), dtype=np.float32)

#for (t) in range(np.shape(low_res)[0]):
#    for i in range(np.shape(low_res)[1]):
#        for j in range(np.shape(low_res)[2]):
#            for k in range(np.shape(low_res)[3]):

#                low_res[t,:,:,:] = data[t*lr_factor[0], i*lr_factor[1], j*lr_factor[2], k*lr_factor[3] ]
#    b = np.broadcast(low_res, shrink(low_res, t))
#    np.copyto(low_res, data, 
#print np.shape(low_res)

# The buffer stores a number of scalar fields
# The buffer index stores the locations of the scalar fields in the buffer
buff_size = np.int32(2)

# Stores the most recently used time slices
buff_index = deque(maxlen=buff_size)
buff_times = []
buff_fields = list(range(buff_size))
for i in range(buff_size):
    buff_index.append(i)
    buff_times.insert(i, 0)
#    mlabscene.children[i] = mlab.pipeline.scalar_field(low_res[0,:,:,:], name='dat'+str(0))
    mlabscene.children[i] = mlab.pipeline.scalar_field(low_res[0,:,:,:])#low_res[0,:,:,:], name='dat'+str(0))
#    mlabscene.children[i].spacing = [1.,lr_factor*20.,1.]
#    mlabscene.children[i].children[0] = mlab.pipeline.volume(mlabscene.children[i])


#print buff_times

class Visualisation(HasTraits):
    time = Range(0,(len(data[:,0,0,0])-1))
    scene = mlabscene
#   plot = Instance(PipelineBase)

    @on_trait_change('time, scene.activated')
    def update_plot(self):
    # If not already loaded:
        if buff_times.count(self.time) == 0:
        # Remove the least recently used index
            n = buff_index.popleft()

        # Clear that whole scene, and clean up
            mlabscene.children[n].remove()
            gc.collect()

            buff_times[n] = self.time
        # Replace the fields, and draw a new volume
            mlabscene.children[n] = mlab.pipeline.scalar_field(low_res[self.time,:,:,:], name='dat'+str(self.time))
#            mlabscene.children[n].spacing = [1.,lr_factor*20.,1.]
            mlabscene.children[n].children[0] = mlab.pipeline.volume(mlabscene.children[n])

        # Hide everything apart from the requested time
            for i in range(buff_size):
                mlabscene.children[i].visible = False
            mlabscene.children[n].visible = True

            buff_index.append(n)

        else:
        # Find which index that time is at
            i = buff_times.index(self.time)
            print i
        # Rotate the recent use index until that's at the front
            n = buff_index[-1]
            while n != i:
                buff_index.rotate(-1)
                n = buff_index[-1]
        
        # Hide everything apart from the requested time
        for f in range(buff_size):
            mlabscene.children[f].visible = False

        # Clear and redraw the requested slice
#        mlabscene.children[n].children[0].remove()
        mlabscene.children[n].children[0] = mlab.pipeline.volume(mlabscene.children[n])
        mlabscene.children[n].visible = True

vis = Visualisation()
vis.configure_traits()
