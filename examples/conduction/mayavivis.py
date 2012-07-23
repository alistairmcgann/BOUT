import numpy as np
from mayavi import mlab
from boutdata import collect

#from boututils import DataFile

from traits.api import HasTraits, Range, Instance, on_trait_change

from traitsui.api import View, Item, Group

from mayavi.core.api import PipelineBase
from mayavi.core.ui.api import MayaviScene, SceneEditor, MlabSceneModel


#Read in the data file, extracting the variable required
#f = DataFile("data/BOUT.dmp.0.nc")
#T = f.read("T")
#f.close()

T = collect("T", path="data")

#t = 0
#Returns the dataset from the variable to be plotted, here for a given time
def T_time(t):
    return T[t,:,:,0]

class Visualisation(HasTraits):
    time = Range(0,(len(T[:,0,0,0])-1))
    scene = Instance(MlabSceneModel, ())
    plot = Instance(PipelineBase)

#    def __init__(self):
#        HasTraits.__init__(self)
#        t = T_time(self.time)
#        self.plot = self.scene.mlab.imshow(T_time(self.time))

    @on_trait_change('time, scene.activated')
    def update_plot(self):
        t = 0 # self.time
        self.plot = self.scene.mlab.imshow(T_time(self.time))
#        if self.plot is None:
#            self.plot = self.scene.mlab.surf(T_time(t = self.time))
#        else:
#            self.plot.mlab_source.t = self.time #set(t=self.time)

    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene), height=250, width=300, show_label=False), Group( '_', 'time', ), resizable=True, )

vis = Visualisation()
vis.configure_traits()

#Plots the dataset for a given timestep
#This timestep should be confined to the shape of the t dimension of T
# =>(T[:,0,0,0])


#t=0
#plot = mlab.imshow(T_time(t))

#Eventually, this will update the plot

#plot.mlab_source.set(t=tnew)
