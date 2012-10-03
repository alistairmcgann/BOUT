import numpy as np
from boututils import DataFile
from boutdata import collect

from threading import Thread
from traits.api import *
from traitsui.api import View, Group, Item, ButtonEditor

var1 = 0.0
var2 = 0.0

class TextDisplay(HasTraits):
    string = String()

    view = View( Item('string', show_label=False, springy=True,style='custom')) 

class LoadVariables(Thread):
    def run(self):
#        self.display.string += 'File' + '\n'
        f = DataFile()
        DataFile.open(f,self.path+self.filename)
        varlist = f.list()
        self.display.string += 'Variables in file :' + '\t' + 'No. of Dims.' + '\n'
        for i in varlist:
            self.display.string += str(i)+ '\t' + str(f.ndims(i))+ '\n'
        f.close()

class MlabPlot(Thread):
    def run(self):
        data1 = collect(self.var1, tind = self.time, path=self.path)
        data2 = collect(self.var2, tind = self.time, path=self.path)
        plot = mlab.figure()
        field1 = mlab.pipeline.scalar_field(data1)
        field2 = mlab.pipeline.scalar_field(data2)
        mlab.pipeline.volume(field1, figure=plot)

        for n in np.around(range(self.no_slices)):
            mlab.pipeline.image_plane_widget(field, figure=plot, plane_orientation = 'y_axes', slice_index=( np.int((n + 3./4.)*((np.shape(data)[1]/self.no_slices)) ) ) )
        

class FilePath(HasTraits):
    display = Instance(TextDisplay)
    load_file = Button()
    path = String()
    filename = String()
    loadvariables = Instance(LoadVariables)
    volume = String()
    slices = String()
    time = CInt()
    no_slices = CInt()
    draw_plot = Button()
    colvar = Instance(MlabPlot)

    view = View( Item('path', style='custom', show_label=True), 
                 Item('filename', style='custom', show_label=True), 
                 Item('load_file', show_label=False),
                 Item('volume', style='custom', show_label=True),
                 Item('slices', style='custom', show_label=True),
                 [Item('time', style='custom', show_label=True),
                 Item('no_slices', style='custom', show_label=True)] ,
                 Item('draw_plot', show_label=False) )

    def _load_file_fired(self):
#        f = DataFile()
#        DataFile.open(f,self.path+self.filename)
#        self.fvars = f.list()
#        f.close()
#        self.display.string += 'Variables loaded :' + '\n'
#        for i in self.fvars:
#            self.display.string += str(i)+ '\n'

#        print self.evar.values
#        self.evar = Enum(fvars)
#        Enum.set_value(evar,fvars)
#        print evar.values

#########################
#Got confused by the different ones, need to swap between loadvariables and mlabplot.

        self.display.string += 'Loading Variables' +'\n'
        self.loadvariables = LoadVariables()
        self.loadvariables.path = self.path
#        self.loadvariables.evar = self.evar
#
        self.loadvariables.start()
#        evar = self.loadvariables.evar
#        print 'g', self.loadvariables.fvars


#        for i in range(len(fvars)):
#        self.evar = 

    def _draw_plot_fired(self):
        self.display.string += 'Collecting variable: ' + self.variablename + '\n'
        self.mlabplot = MlabPlot()
        self.mlabplot.path = self.path

        self.mlabplot.var1 = self.volume
        self.mlabplot.var2 = self.slices
        self.mlabplot.time = self.time
        self.mlabplot.no_slices = self.no_slices
        
        self.colvar.var1 = var1
        self.colvar.var2 = var2
        self.colvar.start()

class LoadFile(HasTraits):
    display = Instance(TextDisplay, ())
    filepath = Instance(FilePath)

    def _filepath_default(self):
        return FilePath(display=self.display)

    view = View( Item('display', style='custom', show_label=False ),
                 Item( 'filepath', style='custom', show_label=False ),
                 resizable=True)

if __name__ == '__main__':
    LoadFile().configure_traits()
