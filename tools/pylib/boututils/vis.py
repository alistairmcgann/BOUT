import numpy as np
from collections import deque
from mayavi import mlab
import gc

from mayavi.api import Engine
#from mayavi.modules.volume import Volume

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
    mlabscene.children[i] = mlab.pipeline.scalar_field(data[0,:,:,:], name='dat'+str(0))
#    buff_fields[i] = mlab.pipeline.scalar_field(data[0,:,:,:], name='dat'+str(0), figure=mlabscene)
print buff_times
#Returns the dataset from the variable to be plotted, here for a given time
def density(t):
    # If not already loaded:
    if buff_times.count(t) == 0:
        # Remove the least recently used index
        n = buff_index.popleft()

        # Clear that whole scene, and clean up
        mlabscene.children[n-buff_size].remove()
#        gc.collect()

        # Insert the new time into the time list
        buff_times[n] = t
        # Create the field and insert it in the fields list
        buff_fields[n] = mlab.pipeline.scalar_field(data[t,:,:,:], name='dat'+str(t) , figure=mlabscene)
#        buff_fields[n].spacing=[1.,20.,1.]

        mlabscene.children[n] = buff_fields[n]


        # Hide everything apart from the requested time
        for i in range(buff_size):
            mlabscene.children[i].visible = False

        mlabscene.children[n].visible = True

#        mlabscene.children[n] = buff_fields[n]
#        mlabscene.add_filter(mlab.pipeline.volume(buff_fields[n]))
        
        # It's now the most recently used index
        buff_index.append(n)
#        return mlabscene.children[n]

        return buff_fields[n]

    else:
        # Find which index that time is at
        i = buff_times.index(t)
        # Rotate the recent use index 'till that's at the front
        n = buff_index[-1]
        while n != i:
            buff_index.rotate()
            n = buff_index[-1]
        
        # Hide everything apart from the requested time
        for i in range(buff_size):
            mlabscene.children[i].visible = False

        mlabscene.children[n].visible = True

    return buff_fields[n]

class Visualisation(HasTraits):
    time = Range(0,(len(data[:,0,0,0])-1))
    scene = mlabscene#Instance(MlabSceneModel, ())
    plot = Instance(PipelineBase)

    @on_trait_change('time, scene.activated')
    def update_plot(self):
        print buff_times.count(self.time)
        print buff_times

    # If not already loaded:
        if buff_times.count(self.time) == 0:
        # Remove the least recently used index
            n = buff_index.popleft()

        # Clear that whole scene, and clean up
            mlabscene.children[n].remove()
            gc.collect()

        # Insert the new time into the time list
            buff_times[n] = self.time

        # Replace the fields
            mlabscene.children[n] = mlab.pipeline.scalar_field(data[self.time,:,:,:], name='dat'+str(self.time))

        # Draw a new volume
            mlabscene.children[n].children[0] = mlab.pipeline.volume(mlabscene.children[n])

        # Hide everything apart from the requested time
            for i in range(buff_size):
                mlabscene.children[i].visible = False

            mlabscene.children[n].visible = True

        # It's now the most recently used index
            buff_index.append(n)

        else:
        # Find which index that time is at
            i = buff_times.index(self.time)
            print i
        # Rotate the recent use index 'till that's at the front
            n = buff_index[-1]
            while n != i:
                buff_index.rotate(-1)
                n = buff_index[-1]
        
        # Hide everything apart from the requested time
        for f in range(buff_size):
            mlabscene.children[f].visible = False

        # Clear and redraw the requested slice
        mlabscene.children[n].children[0].remove()
        mlabscene.children[n].children[0] = mlab.pipeline.volume(mlabscene.children[n])
        mlabscene.children[n].visible = True

#        self.plot = mlab.pipeline.volume(density(self.time))
#        self.plot = mlabscene.children[density(self.time)].pop()

#    mlab.view(figure=mlabscene)
#    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene), height=250, width=300, show_label=False), Group( '_', 'time', ), resizable=True, )

vis = Visualisation()
vis.configure_traits()
