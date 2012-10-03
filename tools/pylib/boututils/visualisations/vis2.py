import numpy as np
from collections import deque
from mayavi import mlab
import gc

from mayavi.api import Engine
#from mayavi.modules.volume import Volume

from traits.api import HasTraits, Range, Instance, on_trait_change, Bool
from traitsui.api import View, Item, Group
from mayavi.core.api import PipelineBase
from mayavi.core.ui.api import MayaviScene, SceneEditor, MlabSceneModel

engine = Engine()
engine.start()
engine.new_scene()
mlabscene = engine.scenes[0]
mlab.show_pipeline()

data = np.load('/hwdisks/data/adm518/d2.npy')

spacing = [1.,20.,1.]

# Generate the low-resolution data
# The factor by which the low-resolution data is reduced by
lr_factor = [1,10,5,10]
low_res = np.ndarray( np.asarray( np.shape(data))[:] / np.asarray(lr_factor)[:], dtype = float)

for t in range(np.shape(low_res)[0]):
    for i in range(np.shape(low_res)[1]):
        for j in range(np.shape(low_res)[2]):
            for k in range( np.shape(low_res)[3] ):

                low_res[t,i,j,k] = data[ t*lr_factor[0],i*lr_factor[1], j*lr_factor[2], k*lr_factor[3] ]

# Initialise the scene
mlabscene.children[0] = mlab.pipeline.scalar_field(data[0,:,:,:], name='dat'+str(0))
mlabscene.children[0].spacing = spacing #[1.,20.,1.]
mlabscene.children[0].children[0] = mlab.pipeline.volume(mlabscene.children[0])

class Visualisation(HasTraits):
    time = Range(0,(len(data[:,0,0,0])-1))
    scene = mlabscene
    high_res = Bool(False)

    @on_trait_change('time, scene.activated, high_res')
    def update_plot(self):
        # Choose the resolution based on state of high_res
        if self.high_res == False :
            # Delete the whole scene
            mlabscene.children[0].remove()
            # Replace the fields
            mlabscene.children[0] = mlab.pipeline.scalar_field(low_res[self.time,:,:,:], name='dat'+str(self.time))
            mlabscene.children[0].spacing = [ spacing[0] * lr_factor[1], spacing[1] * lr_factor[2], spacing[2] * lr_factor[3] ]
            mlabscene.children[0].children[0] = mlab.pipeline.volume(mlabscene.children[0])
            gc.collect()

        elif self.high_res == True:
            # Delete the whole scene
            mlabscene.children[0].remove()
        # Replace the fields
            mlabscene.children[0] = mlab.pipeline.scalar_field(data[self.time,:,:,:], name='dat'+str(self.time))
            mlabscene.children[0].spacing = spacing#[1.,20.,1.]
            mlabscene.children[0].children[0] = mlab.pipeline.volume(mlabscene.children[0])
            gc.collect()
            
vis = Visualisation()
vis.configure_traits()
