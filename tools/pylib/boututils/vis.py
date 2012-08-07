import numpy as np
from collections import deque
from mayavi import mlab
import gc

from mayavi.api import Engine
from traits.api import HasTraits, Range, Instance, on_trait_change, Bool, Button
from traitsui.api import View, Item, Group
from mayavi.core.api import PipelineBase
from mayavi.core.ui.api import MayaviScene, SceneEditor, MlabSceneModel

engine = Engine()
engine.start()
engine.new_scene()
mlabscene = engine.scenes[0]
mlab.show_pipeline()

data = np.load('/home/adm518/Documents/dtime.npy')

lr_factor = [1,10,5,10]

low_res = np.ndarray(np.asarray(np.shape(data))[:] / np.asarray(lr_factor)[:], dtype=float)

for t in range(np.shape(low_res)[0]):
    for i in range(np.shape(low_res)[1]):
        for j in range(np.shape(low_res)[2]):
            for k in range( np.shape(low_res)[3] ):

                low_res[t,i,j,k] = data[ t*lr_factor[0],i*lr_factor[1], j*lr_factor[2], k*lr_factor[3] ]

buff_size = np.int32(2)

# Stores the most recently used time slices
buff_index = deque(maxlen=buff_size)
buff_times = []
buff_fields = list(range(buff_size))
for i in range(buff_size):
    buff_index.append(i)
    buff_times.insert(i, 0)
    mlabscene.children[i] = mlab.pipeline.scalar_field(low_res[0,:,:,:])
    mlabscene.children[i].spacing = [1.*lr_factor[1],20.*lr_factor[2],1.*lr_factor[3]]

class Visualisation(HasTraits):
    time = Range(0,(len(data[:,0,0,0])-1))
    scene = mlabscene
    high_def = Bool(False)

    @on_trait_change('time, scene.activated, high_def')
    def update_plot(self):
    # If not already loaded:
        if buff_times.count(self.time) == 0:
        # Remove the least recently used index
            n = buff_index.popleft()

        # Clear that whole scene, and clean up
            mlabscene.children[n].remove()
            gc.collect()

            buff_times[n] = self.time

            if self.high_def == False:
                # Replace the fields, and draw a new volume
                mlabscene.children[n] = mlab.pipeline.scalar_field(low_res[self.time,:,:,:], name='dat'+str(self.time))
                mlabscene.children[n].spacing = [1.*lr_factor[1],20.*lr_factor[2],1.*lr_factor[3]]
                mlabscene.children[n].children[0] = mlab.pipeline.volume(mlabscene.children[n])
            elif self.high_def == True:
                # Replace the fields, and draw a new volume
                mlabscene.children[n] = mlab.pipeline.scalar_field(data[self.time,:,:,:], name='dat'+str(self.time))
                mlabscene.children[n].spacing = [1.,20.,1.]
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
        mlabscene.children[n].visible = True
        mlabscene.children[n].children[0] = mlab.pipeline.volume(mlabscene.children[n])


vis = Visualisation()
vis.configure_traits()
