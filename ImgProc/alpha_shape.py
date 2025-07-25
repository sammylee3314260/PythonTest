import numpy as np
import pyvista as pv
import tifffile

def alpha_shape(points = None,alpha:int = 0.3):
    if points is None:
        points = np.random.rand(100, 3) # meaning 100 random points in 3D space
    if points.dtype != 'float': points = np.float32(points)
    print(points.shape)
    print("cloud")
    cloud = pv.PolyData(points)
    print("tetra")
    tetra = cloud.delaunay_3d(alpha=alpha)
    print("shell")
    shell = tetra.extract_geometry()
    print("plotter")
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