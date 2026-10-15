### 
# This script is specific for multi-channel Z stack images, it will:
# - Read czi files.
# - Change to individual channel tif and mp4 files.
# - max_projection to tif and jpeg files.
from typing import Final, Tuple, Dict, Generator
PREFIX_SPLITTER: Final[str|None] = None
import sys
from pathlib import Path
import aicspylibczi
from itertools import product
import tifffile
import numpy as np


if __name__ == "__main__" and __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    __package__ = 'ImgProc'
from . import utils
def iter_multi_axes(axes:str = 'BVCTZYX0',# = "",
                    shape:Tuple[int,...] = (1, 1, 4, 1, 16, 1024, 1024, 1),# = None,
                    iter_axes:Dict[str,...] = {'S','T','C'},
                    pres_axes:Dict[str,...] = {'X','Y'},
                    file_pref:str = "img_") ->(
                    Generator[tuple[tuple[int|None,...],int|None,str,str],None,None]):
    axes_map = {ax:i for i,ax in enumerate(axes)}
    iter_axes = set(iter_axes)
    pres_axes = set(pres_axes)
    print("before set manipulation: ",iter_axes, pres_axes)
    # check axes for filename
    is_multi_scene, is_multi_time, is_multi_channel, is_Zstack = False,False,False, False
    if 'S' in axes_map and shape[axes_map['S']] > 1: is_multi_scene = True
    if 'T' in axes_map and shape[axes_map['T']] > 1: is_multi_time = True
    if 'C' in axes_map and shape[axes_map['C']] > 1: is_multi_channel = True
    if 'Z' in axes_map and shape[axes_map['Z']] > 1: is_Zstack = True
    axes_str = 'YX'
    if is_Zstack:
        pres_axes.add('Z')
        axes_str = 'Z'+axes_str
    elif is_multi_time:
        iter_axes.remove('T')
        pres_axes.add('T')
        axes_str = 'T'+axes_str
    print("after set manipulation: ", iter_axes, pres_axes)

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
    if is_Zstack: # 'Z' in axes_map and 'Z' in pres_axes: # check if this img is 3D, if so return new index of Z
        cnt = 0
        for ax in axes_map:
            if ax == 'Z': Z_ind = cnt; break
            elif ax in pres_axes: cnt += 1

    for index_combo in product(*loop_range):
        slicer = [0] * len(shape) # index = dimension of the img
        for ax, idx in zip(loop_axes, index_combo): slicer[axes_map[ax]] = idx
        for ax in pres_axes:
            if ax in axes_map and not ax in loop_axes: slicer[axes_map[ax]] = slice(None)
            # This is a very important line, slice(None) = [:] so we can take the whole stack!
        out_filename = (file_pref[0:-1]
                        + ('_S' + str(slicer[axes_map.get('S')]) if is_multi_scene else '')
                        + ('_T' + str(slicer[axes_map.get('T')]) if is_multi_time and 'T' in iter_axes else '')
                        + ('_C' + str(slicer[axes_map.get('C')]) if is_multi_channel else '')
                        )
        out_file_max = (file_pref + 'max'
                        + ('_S' + str(slicer[axes_map.get('S')]) if is_multi_scene else '')
                        + ('_T' + str(slicer[axes_map.get('T')]) if is_multi_time and 'T' in iter_axes else '')
                        + ('_C' + str(slicer[axes_map.get('C')]) if is_multi_channel else '')
                        )
        yield tuple(slicer), Z_ind, out_filename, out_file_max, axes_str

def tiffw(path, img_array, axes, shape, file_pref,scalefactor):
    ZSTACK_OUT = Path(path) / 'z_stack'
    MAX_PROJ_OUT = Path(path) / 'max_proj'
    if not ZSTACK_OUT.exists(): ZSTACK_OUT.mkdir()
    if not MAX_PROJ_OUT.exists(): MAX_PROJ_OUT.mkdir()
    if file_pref[-1]!='_': file_pref = file_pref+'_' # this might have to be fine tuned
    for ind, Z_ind, out_filename, out_file_max, axes_str in iter_multi_axes(axes,shape,file_pref=file_pref):
        print(out_filename)
        img = img_array[ind]
        print(img.shape)
        # check scale factor
        '''
        resolution manipulation:
            resolution code: X: 282, Y:283
            in czi file the resolution unit is meter/pixel
            in tiff file the resolution unit is pixel/resUnit (Code 296)
        '''
        tifffile.imwrite(ZSTACK_OUT / (out_filename + '.tif'), img,
                        shape=img.shape,
                        imagej=True,
                        dtype=img.dtype,
                        software='ImageJ',
                        resolution=((int(1 / scalefactor['Y']), int(1e6)), (int(1 / scalefactor['X']), int(1e6))),
                        metadata={
                            'spacing': scalefactor['Z'] * int(1e6),
                            'unit': 'micron',
                            'axes': axes_str})
        # max projection
        if (Z_ind is not None) and (img.ndim >= 3):
            maxproj = np.max(img, axis = Z_ind)
            tifffile.imwrite(MAX_PROJ_OUT / (out_file_max + '.tif'), maxproj,
                             shape=maxproj.shape,
                             imagej=True,
                             dtype=maxproj.dtype,
                             software='ImageJ',
                             resolution=((int(1 / scalefactor['Y']), int(1e6)), (int(1 / scalefactor['X']), int(1e6))),
                             # unit px/m
                             metadata={
                                 'unit': 'micron',
                                 'axes': 'YX'})

def main():
    # main function
    if sys.argv[1:]: parent_folder_list = sys.argv[1:]
    else:
        print(f"Enter file/folder input section, you can also drag the file/folders to this app.")
        parent_folder = utils.get_filepath()
        parent_folder_list = [parent_folder]
    print(parent_folder_list)
    for parent_folder in parent_folder_list:
        print(f"parent_folder: {parent_folder}")
        if type(parent_folder) != str or parent_folder == "": print(f"The provided path does not exist: {parent_folder}", file=sys.stderr); continue
        path_parent_folder = Path(parent_folder).expanduser()
        if not path_parent_folder.exists(): print(f"The provided path does not exist: {path_parent_folder}", file=sys.stderr); continue
        czi_files = []
        if path_parent_folder.is_file() and (path_parent_folder.suffix.lower() == '.czi'):
            print(f"The provided path is a czi file: {path_parent_folder}")
            czi_files.append(path_parent_folder)
            path_parent_folder = path_parent_folder.parent
        elif path_parent_folder.is_dir():
            print(f"The provided path is a directory: {path_parent_folder}")
            czi_files = sorted((p for p in path_parent_folder.glob('*.czi')),
                                    key=lambda p: utils.natural_sort_key(p.name))
            if len(czi_files) == 0: print(f"No czi files found in this dir {path_parent_folder}", file=sys.stderr); continue
        for f in czi_files: # f should be poxis Path object!
            print(f)
            cziimg = aicspylibczi.CziFile(str(f))
            shape = cziimg.size
            if shape == (0,0): print(f"This file {str(f)} has shape (0,0).\nThe file might be a main file of a group of split files. Cannot read!\n"); continue
            axes = cziimg.dims
            print(f.name,shape,axes)
            scaling = cziimg.meta.find(".//Scaling")
            scalefactor = {}
            if scaling is not None:
                for dist in scaling.findall("Items/Distance"):
                    axis = dist.attrib.get("Id")
                    val = float(dist.find("Value").text)
                    print(dist.attrib)
                    # unit = dist.find("DefaultUnitFormat").text
                    # scalefactor[axis] = (val,unit)
                    scalefactor[axis] = val
                    # print(axis, val, unit) # axes: 'X' 'Y' 'Z'
                    print(axis, val)
            img_array, _ = cziimg.read_image()
            if isinstance(PREFIX_SPLITTER, str): prefix_splitter = PREFIX_SPLITTER
            else: prefix_splitter = '.czi'
            file_pref = f.name.split(prefix_splitter)[0]
            tiffw(f.parent, img_array, axes, shape, file_pref, scalefactor)

if __name__ == "__main__":
    main()