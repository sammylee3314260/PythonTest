import numpy as np
import pyvista as pv
import tifffile

def alpha_shape(points = None,alpha:int = 0.1):
    if points==None:
        points = np.random.rand(100, 3) # meaning 100 random points in 3D space
    cloud = pv.PolyData(points)
    tetra = cloud.delaunay_3D(alpha=alpha)
    shell = tetra.extract_geometry()

    if 1: # to plot
        plotter = pv.Plotter()
        plotter.add_mesh(shell,color='lightblue',opacity=0.6)
        plotter.add_points(points,color='red',point_size=5)
        plotter.show()

if __name__ == '__main__':
    path = ''
    file = ''
    if path == '':
        alpha_shape()

