### 
# This script is specific for multi-channel Z stack images, it will:
# - Read czi files.
# - Change to individual channel tif and mp4 files.
# - max_projection to tif and jpeg files.
from typing import Final
PREFIX_SPLITTER: Final[str|None] = None
# You can change this to seperate out "Prefix" of the file.
# For example, for LSM 880 you might have '{your file name}_2026_10_15__21_31_06.czi' as a file name,
# you can use '_2026_10_15_" to get rid of everything after your desired file name.

import sys
from pathlib import Path
import aicspylibczi
import tifffile
import numpy as np

if __name__ == "__main__" and __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    __package__ = 'ImgProc'
from . import utils
from . import image_utils

def tiffw(path, img_array, axes, shape, file_pref,scalefactor):
    ZSTACK_OUT = Path(path) / 'z_stack'
    MAX_PROJ_OUT = Path(path) / 'max_proj'
    if not ZSTACK_OUT.exists(): ZSTACK_OUT.mkdir()
    if not MAX_PROJ_OUT.exists(): MAX_PROJ_OUT.mkdir()
    if file_pref[-1]!='_': file_pref = file_pref+'_' # this might have to be fine tuned
    for ind, Z_ind, out_filename, out_file_max, axes_str in image_utils.iter_multi_axes(axes,shape,file_pref=file_pref):
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
        # Z stack save as tif
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
        # can also save as mp4 w/ overlaying scalebar

        # max projection save as tif
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
        # can also save as jpeg w/ overlaying scalebar

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
    input("Press Enter to close...")

if __name__ == "__main__":
    main()