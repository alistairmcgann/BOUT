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

data = np.load('./dtime.npy')

# The buffer stores a number of scalar fields
# The buffer index stores the locations of the scalar fields in the buffer
buff_size = np.int32(2)

# Stores the most recently used time slices
buff_index = deque(maxlen=buff_size)
buff_times = []
buff_fields = list(range(buff_size))
for i in range(buff_size):
    buff_index.append(i)
    buff_times.insert(0, i)
    buff_fields[i] = mlab.pipeline.volume(mlab.pipeline.scalar_field(data[0,:,:,:], name='dat'+str(0)), figure=mlabscene)

#Returns the dataset from the variable to be plotted, here for a given time
def density(t):
    # If not already loaded:
    if buff_times.count(t) == 0:
        # Remove the least recently used index
        n = buff_index.popleft()

        mlabscene.children.remove('dat'+str(buff_times[n]))
        gc.collect()

        # Insert the new time into the time list
        buff_times[n] = t
        # Create the field and insert it in the fields list
        buff_fields[n] = mlab.pipeline.volume(mlab.pipeline.scalar_field(data[t,:,:,:], name='dat'+str(t) ), figure=mlabscene)

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
        
#        return n

    return buff_fields[n]

class Visualisation(HasTraits):
    time = Range(0,(len(data[:,0,0,0])-1))
    scene = mlabscene#Instance(MlabSceneModel, ())
    plot = Instance(PipelineBase)

    @on_trait_change('time, scene.activated')

#    density(self.time, self)

    def update_plot(self):
#        self.plot = mlab.pipeline.volume(density(self.time), figure=self.scene)
        self.plot = density(self.time)
#        self.plot = mlabscene.children[density(self.time)].pop()

    mlab.view(figure=mlabscene)
#    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene), height=250, width=300, show_label=False), Group( '_', 'time', ), resizable=True, )

vis = Visualisation()
vis.configure_traits()
