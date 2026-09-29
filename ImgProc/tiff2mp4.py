###
# This code is to save tiff into mp4 because imageJ only save to avi.

import tifffile
import sys
from pathlib import Path
import numpy as np
import warnings
if __name__ == "__main__" and __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    __package__ = 'ImgProc'
from . import utils

### MP4 parameters
import cv2
# height for movie resizing
COMPRESSION = True   # Whether you want compression
DESIRED_HEIGHT = 480 # if you want compression, what is the targeted img height in px
# FPS for mp4 file (play rate)
OUTPUT_FPS = 10
# Scale bar target length in µm
SCALEBAR_UM = 250   # Default 50 µm
# Font setting
FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
FONT_COLOR = (255, 255, 255)   # white
FONT_THICK = 2
# BG_COLOR = FONT_COLOR
BG_COLOR   = (0, 0, 0)         # Black Bg frame
### End of parameters

def main():
    if sys.argv[1:]: parent_folder_list = sys.argv[1:]
    else:
        print(f"Enter file/folder input section, you can also drag the file/folders to this app.")
        parent_folder = utils.get_filepath()
        parent_folder_list = [parent_folder]
    # parent_folder_list = ["/mnt/c/Users/l.ping-hsien/Desktop/0.pre/2026-06-10/outputs/tiff/"] #utils.get_filepath(try_gui=True, sys_argv=sys.argv)
    for parent_folder in parent_folder_list:
        print(f"parent_folder: {parent_folder}")
        if type(parent_folder) != str or parent_folder == "": print(f"The provided path does not exist: {parent_folder}", file=sys.stderr); continue
        path_parent_folder = Path(parent_folder).expanduser()
        if not path_parent_folder.exists(): print(f"The provided path does not exist: {path_parent_folder}", file=sys.stderr); continue
        tiff_files = []
        if path_parent_folder.is_file() and (path_parent_folder.suffix in ('.tif','.tiff')):
            print(f"The provided path is a tif(f) file: {path_parent_folder}")
            tiff_files.append(path_parent_folder)
            path_parent_folder = path_parent_folder.parent
        elif path_parent_folder.is_dir():
            print(f"The provided path is a directory: {path_parent_folder}")
            tiff_files = sorted((p for p in path_parent_folder.iterdir()
                                 if p.suffix.lower() in ('.tif', '.tiff')),
                                 key=lambda p: utils.natural_sort_key(p.name)) # O(n) scan dir to compare tif/tiff/TIF/TIFF ext varient
            if len(tiff_files) == 0: print(f"No tif(f) files found in this dir {path_parent_folder}", file=sys.stderr); continue
        else: print(f"The providetifd path is neither a tif(f) file nor a directory: {path_parent_folder}",file=sys.stderr); continue
        # for i in tiff_files: print(i)
        for f in tiff_files: # f should be poxis Path object!
            print(f)
            img = None
            with tifffile.TiffFile(str(f)) as tif:
                img = tif.asarray() # Now img is numpy ndarray object
                axes = tif.series[0].axes
            if 'X' not in axes or 'Y' not in axes:
                # raise ValueError(f"Axes {axes} lack X or Y.")
                print(f"File {f} not convertible, Axes {axes} lack X or Y.",file=sys.stderr); continue
            if 'C' in axes:
                # raise ValueError(f"Please convert multichannel image to RGB images in ImageJ first. Current axes {axes}.")
                print(f"File {f} not convertible, Please convert multichannel image to RGB images in ImageJ first. Current axes {axes}.",file=sys.stderr);continue

            # This is just a protection to prevent iterating the wrong axis.
            spatial = [a for a in ('Y','X','S') if a in axes]
            seq = [a for a in axes if a not in spatial]
            if len(seq) > 1: warnings.warn(f"Warning: File {f} non spatial axes more than 1: axes = {axes}. Do you really mean it?")
            order = [axes.index(a) for a in seq] + [axes.index(a) for a in spatial]
            img = np.transpose(img, order)
            img = img.reshape(-1, *img.shape[len(seq):]) if seq else img[np.newaxis,...]

            is_multicolor = None
            n_samples = None
            if 'S' in axes:
                # The image should be multicolor, S should be 3
                is_multicolor = True
                n_samples = img.shape[axes.index("S")]
                if n_samples not in (3,4):print(f"File {f} Unknown number of samples: {n_samples}. Axes is: {axes}",file=sys.stderr);continue
                    # raise ValueError(f"Unknown number of samples: {n_samples}. Axes is: {axes}")
            else: is_multicolor = False

            scale_factor = DESIRED_HEIGHT / img.shape[1] if COMPRESSION else 1
            out = None
            for slice in img:
                if is_multicolor:
                    if n_samples == 3: frame_bgr = cv2.cvtColor(slice, cv2.COLOR_RGB2BGR)
                    elif n_samples == 4: frame_bgr = cv2.cvtColor(slice, cv2.COLOR_RGBA2BGR)
                    else: print(f"File {f} Unknown number of channels: {n_samples}. Axes is: {axes}. Second check. Plz debug why the first check didnt find it.",file=sys.stderr); continue
                        # raise ValueError(f"Unknown number of channels: {n_samples}. Axes is: {axes}. Second check. Plz debug why the first check didnt find it.")
                else: frame_bgr = cv2.cvtColor(slice, cv2.COLOR_GRAY2BGR)
                resized_slice = cv2.resize(frame_bgr,dsize=None,fx=scale_factor,fy=scale_factor,interpolation=cv2.INTER_LINEAR)
                if out is None:
                    out = cv2.VideoWriter(str(path_parent_folder / (f.stem + '.mp4')),
                                        cv2.VideoWriter_fourcc(*'mp4v'),
                                        OUTPUT_FPS,
                                        (resized_slice.shape[1],resized_slice.shape[0])
                                        )
                out.write(resized_slice)
            out.release()
            print(f"File {f} Output @ {str(path_parent_folder / (f.stem + '.mp4'))}")
    input("Press Enter to continue...")
if __name__ == "__main__":
    main()