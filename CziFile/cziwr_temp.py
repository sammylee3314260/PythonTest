import czifile
from itertools import product
from typing import Tuple, Generator

def iter_multi_axes(axes:str,
                    shape:Tuple[int],
                    iter_axes = {'S','T','C'},
                    pres_axes = {'X','Y','Z'},
                    file_pref = "img_") -> (
                        Generator[tuple[tuple[slice | int, ...], int | None , str],None,None]):
    
    axes_map = {ax:i for i, ax in enumerate(axes)}
    # check channels for filenames
    is_multi_time, is_multi_channel,is_multi_scene = False, False, False
    if 'T' in axes_map and shape()[axes_map['T']]!=1: is_multi_time = True
    if 'C' in axes_map and shape()[axes_map['C']]!=1: is_multi_channel = True
    if 'S' in axes_map and shape()[axes_map['S']]!=1: is_multi_scene = True

    # check whether the intend-to-iterate axes are in the shape,
    #  if so add them into to-be-iterated list
    loop_axes = []
    loop_range = []
    for ax in iter_axes:
        if ax in axes_map:
            loop_axes.append(ax)
            loop_range.append(range(shape[axes_map[ax]]))
    Z_ind = None
    if 'Z' in axes_map and 'Z' in pres_axes:
        Z_ind = axes_map['Z']
    for index_combo in product(*loop_range):
        slicer = [0] * len(shape) # len(shape) is the dimension of the image

        for ax,idx in zip(loop_axes,index_combo):
            slicer[axes_map[ax]] = idx
        for ax in pres_axes:
            if ax in axes_map: slicer[axes_map[ax]] = slice(None)
        out_filename = (file_pref
                        + ('S'+ str(index_combo[axes_map['S']]) if is_multi_scene else '')
                        + ('T'+ str(index_combo[axes_map['T']]) if is_multi_time else '')
                        + ('C'+ str(index_combo[axes_map['C']]) if is_multi_channel else '')
        )
        yield tuple(slicer), Z_ind, out_filename

path = 'czis/'
file = 'prefix_405_nuc'
with czifile.CziFile(path+file) as cziimg:
    img = cziimg.asarray()
    for ind, Z_ind, out_filename in iter_multi_axes(cziimg.axes,cziimg.shape,file_pref=file.split('405')[0]):
        img_cur = img[ind]
        if (Z_ind == None) != (img_cur.shape == 2):
            print("Not 2D img but no Z returned??")
            exit()
        elif Z_ind == None and img_cur.ndim == 2:
            img_max = img_cur
        else:
            img_max = max(img, axes = Z_ind)