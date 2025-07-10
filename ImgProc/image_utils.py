############################
# image processing tools
# - czi to tif
# - tif to jpeg
# - tif to npy
# - npy + mask to quant
############################

import numpy as np
import czifile
import tifffile
import os
from typing import Tuple,Dict,Generator, List
from itertools import product
import matplotlib.pyplot as plt

def iter_multi_axes(axes:str = 'BVCTZYX0',# = "",
                    shape:Tuple[int,...] = (1, 1, 4, 1, 16, 1024, 1024, 1),# = None,
                    iter_axes:Dict[str,...] = {'S','T','C'},
                    pres_axes:Dict[str,...] = {'X','Y','Z'},
                    file_pref:str = "img_") ->(
                    Generator[tuple[tuple[int|None,...],int|None,str,str],None,None]):

    axes_map = {ax:i for i,ax in enumerate(axes)}
    # check axes for filename
    is_multi_scene, is_multi_time, is_multi_channel = False,False,False
    if 'S' in axes_map and shape[axes_map['S']] > 1: is_multi_scene = True
    if 'T' in axes_map and shape[axes_map['T']] > 1: is_multi_time = True
    if 'C' in axes_map and shape[axes_map['C']] > 1: is_multi_channel = True

    # check if the to-be-iterated axes exist,
    # if so add to iter_axes
    loop_axes = []
    loop_range = []
    for ax in iter_axes:
        if ax in axes_map:
            print(ax, shape[axes_map[ax]])
            loop_axes.append(ax)
            loop_range.append(range(shape[axes_map[ax]]))
    Z_ind = None
    if 'Z' in axes_map and 'Z' in pres_axes: # check if this img is 3D, if so return index of Z
        pres_ind = []
        cnt = 0
        for ax in axes_map:
            if ax == 'Z':
                Z_ind = cnt
                break
            elif ax in pres_axes:
                cnt += 1

    for index_combo in product(*loop_range):
        slicer = [0] * len(shape) # index = dimension of the img

        for ax, idx in zip(loop_axes, index_combo):
            slicer[axes_map[ax]] = idx
        for ax in pres_axes:
            if ax in axes_map and not ax in loop_axes: slicer[axes_map[ax]] = slice(None)
        out_filename = (file_pref
                        + ('S' + str(slicer[axes_map.get('S')]) if is_multi_scene else '')
                        + ('T' + str(slicer[axes_map.get('T')]) if is_multi_time else '')
                        + ('C' + str(slicer[axes_map.get('C')]) if is_multi_channel else '')
                        )
        out_file_max = (file_pref + 'max_'
                        + ('S' + str(slicer[axes_map.get('S')]) if is_multi_scene else '')
                        + ('T' + str(slicer[axes_map.get('T')]) if is_multi_time else '')
                        + ('C' + str(slicer[axes_map.get('C')]) if is_multi_channel else '')
                        )
        yield tuple(slicer), Z_ind, out_filename, out_file_max

def cziwr(path:str = '', filelist:List[str] = []):
    if not os.path.exists(os.path.join(path,'tiff/')):
        os.mkdir(os.path.join(path,'tiff/'))
    for file in filelist:
        if not os.path.exists(os.path.join(path, file)):
            print(f'File {file} does not exist')
            continue
        if not file.lower().endswith('.czi'):
            print(f'File {file} is not a czi file')
            continue
        # axes, shape, img_array, file_pref, scalefactorX, scalefactorY, scalefactorZ = '',(), None,'',0,0,0
        with czifile.CziFile(os.path.join(path,file)) as cziimg:
            print(cziimg.axes, cziimg.shape)  # (1, 1, 4, 1, 17, 1024, 1024, 1)
            # What I usually care: scene(S), time (T), channel (C), x, y, z
            # cziimg.axes = 'BVSCTZYX0'
            # cziimg.shape = (1, 1, 2, 1, 3, 34, 1000, 1000, 1)
            axes = cziimg.axes
            shape = cziimg.shape
            file_pref = file.split('2025')[0] # this might have to be fine tuned
            scalefactorstrings = cziimg.metadata().split('<Distance Id="X">')[1].split('</Distance>')
            scalefactorX = float(scalefactorstrings[0].split('</Value>')[0].split('<Value>')[1])
            scalefactorY = float(scalefactorstrings[1].split('</Value>')[0].split('<Value>')[1])
            scalefactorZ = float(scalefactorstrings[2].split('</Value>')[0].split('<Value>')[1])
            img_array = cziimg.asarray()
        for ind, Z_ind, out_filename, out_file_max in iter_multi_axes(axes,shape,file_pref=file_pref):
            img = img_array[ind]
            # check scale factor
            '''
            resolution manipulation:
                resolution code: X: 282, Y:283
                in czi file the resolution unit is meter/pixel
                in tiff file the resolution unit is pixel/resUnit (Code 296)
            '''
            tifffile.imwrite(path + 'tiff/' + out_filename + '.tif', img,
                            shape=img.shape,
                            imagej=True,
                            dtype=img.dtype,
                            software='ImageJ',
                            resolution=((int(1 / scalefactorY), 1000000), (int(1 / scalefactorX), 1000000)),
                            metadata={
                                'spacing': scalefactorZ * 1000000,
                                'unit': 'micron',
                                'axes': 'ZYX'})
            # max projection
            if (Z_ind is not None) and (img.ndim >= 3):
                maxproj = np.max(img, axis = Z_ind)
                tifffile.imwrite(path + 'tiff\\' + out_file_max + '.tif', maxproj,
                                 shape=maxproj.shape,
                                 imagej=True,
                                 dtype=maxproj.dtype,
                                 software='ImageJ',
                                 resolution=((int(1 / scalefactorY), 1000000), (int(1 / scalefactorX), 1000000)),
                                 # unit px/m
                                 metadata={
                                     'unit': 'micron',
                                     'axes': 'YX'})

#def tiff2jpeg(path = '', filelist = []):
#    if not os.path.exists(path+'tiff/'):

if __name__ == '__main__':
    path = 'C:\\Users\\pathaklab\\Box\\HyperOsmo\\YAP_Actin_lamAC\\'
    filnames = os.listdir(path)
    cziwr(path, filnames)