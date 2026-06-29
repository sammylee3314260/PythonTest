###
# The purpose of this code is to:
# 1. Read time-lapse czi files, should be 3 layer z stack with time-lapse of difference time duration.
# 2. Pick the best focal plane.
# 3. Output to (can choose): (1) tiff. (2) npy (image data) + json (metadata). (3) mp4.

from typing import Final
### Parameters
# These are constant parameters for you to set as you need before run.
# The Final[] thing is just for type hinting, so that I will not accidentally change them during the code is running.
DEBUG: Final[bool] = True # If true, the path will take debug path (testing env)
IMG_DISPLAY: Final[bool] = False # If true, import mayplotlib and show all three z images to pick
# Input folder path input
INPUT_FOLDER_PATH: Final[str|None] = "/mnt/f/Osmolarity/2026-06-10/2.recovery1" # If not None, use this path as the input folder path, otherwise ask user to select a folder. Only used when DEBUG is False.
# Output folders path input
OUTPUT_FOLDER: str|None = "/mnt/d/Osmolarity/260610/2.recovery1/"            # Will be changed if None
OUTPUT_TIFF: str|None = None              # output folder of tiff files, Will be changed if None
OUTPUT_NPY: str|None = None               # output folder of npy/json files, Will be changed if None
OUTPUT_MP4: str|None = None               # output folder of mp4 files, Will be changed if None
# Metadata parameters (for movie overlay mostly)
PIXEL_SIZE: float|None = None
TIME_UNIT: Final[str] = "min"       # I guess we should always use min, because PIV analysis use min??? dk
PLOT_TIME_UNIT: Final[str] = "min"  # You can change this one, Time unit for plotting.
TIME_STEP: float|None = None
# Save Parameters
SAVE_TIFF: Final[bool] = True      # Whether you want tifffile
SAVE_NPY: Final[bool] = True       # Whether you want npy/json
SAVE_MP4: Final[bool] = True       # Whether you want mp4
### MP4 parameters
import cv2
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

import os
import sys
import glob
from pathlib import Path
import re
from aicspylibczi import CziFile
import numpy as np
if IMG_DISPLAY: import matplotlib.pyplot as plt


if SAVE_TIFF:
    import tifffile
if SAVE_NPY:
    import json

def get_filepath(try_gui:bool = True):
    if sys.argv[1:]: return sys.argv[1]
    if INPUT_FOLDER_PATH: return INPUT_FOLDER_PATH
    import subprocess
    cmd = [
        "powershell.exe",
        "-NoProfile",
        "-Command",
        "& { $p = New-Object -ComObject Shell.Application; " +
        "$f = $p.BrowseForFolder(0, 'Select a folder', 0); " +
        "if ($f) { $f.Self.Path } }"
    ]
    if try_gui:
        try:
            win_path = subprocess.check_output(cmd).decode('utf-8').strip()
            if win_path:
                unix_path = subprocess.check_output(["wslpath", "-u", win_path]).decode('utf-8').strip()
                return unix_path
        except subprocess.CalledProcessError:
            print("Cannot open windows file explorer. Try cli.")
    import readline
    
    readline.set_completer_delims(' \t\n;')
    readline.parse_and_bind("tab: complete")
    def path_completer(text, state):
        return (glob.glob(os.path.expanduser(text) + '*') + [None])[state]
    readline.set_completer(path_completer)
    parent_folder = input("Please enter the path to the parent folder, you can use tab:\n")
    return parent_folder

def natural_sort_key(s):
    """sort filename (e.g. _2_ < _10_)"""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

def normalize_frame(frame: np.ndarray) -> np.ndarray:
    # if frame.dtype == np.uint8: return frame
    f_min, f_max = np.percentile(frame, (1,99))
    # f_min, f_max = frame.min(), frame.max()
    if f_max == f_min:
        return np.zeros_like(frame, dtype=np.uint8)
    norm = (frame.astype(np.float32) - f_min) / (f_max - f_min) * 255
    return norm.astype(np.uint8)

def best_focus_z(czi, t, z_planes):
    '''
    Akash's matlab code for best focal plane selection:

    % Method 1: Variance of Laplacian (edge strength)
    laplacianKernel = [0 1 0; 1 -4 1; 0 1 0];
    imgLaplacian = abs(conv2(double(img), laplacianKernel, 'same'));
    focusMetrics(i) = var(imgLaplacian(:));

    % Method 2: Gradient magnitude
    [gx, gy] = gradient(double(img));
    gradMag = sqrt(gx.^2 + gy.^2);
    gradMetric = mean(gradMag(:));

    % Method 3: Normalized variance
    imgNorm = double(img) / 255;
    normVar = var(imgNorm(:)) / (mean(imgNorm(:)) + eps);

    % Combine metrics
    focusMetrics(i) = focusMetrics(i) + gradMetric + normVar;
    '''
    scores_lap = []
    scores_grad = []
    scores_gaus = []
    for z in range(z_planes):
        frame, _ = czi.read_image(T=t, Z=z, C=0)
        frame = normalize_frame(np.squeeze(frame))
        # frame = np.squeeze(frame)
        frame_gaus = cv2.GaussianBlur(frame, (3,3), 0)
        # Laplacian
        if IMG_DISPLAY:
            plt.subplot(2, z_planes, z+1)
            plt.imshow(frame_gaus)
            plt.title(f'Z = {z}')
        score_lap = cv2.Laplacian(frame_gaus, cv2.CV_64F).var()
        scores_lap.append(score_lap)

        # Tenegrad
        gx = cv2.Sobel(frame_gaus, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(frame_gaus, cv2.CV_64F, 0, 1, ksize=3)
        g3 = np.hypot(gx,gy)
        if IMG_DISPLAY:
            plt.subplot(2, z_planes, 3 + z+1)
            plt.imshow(g3)
        scores_grad.append(g3.mean())

        # Gaussian ratio
        frame_gaus = cv2.GaussianBlur(frame, (1,1), 0)
        gx = cv2.Sobel(frame_gaus, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(frame_gaus, cv2.CV_64F, 0, 1, ksize=3)
        g1 = np.hypot(gx,gy)
        scores_gaus.append(g1.mean()/scores_grad[z])

    print(f"Lap score = {scores_lap[:]}\nGrad score = {scores_grad[:]}\nGauss score = {scores_gaus[:]}")
    if IMG_DISPLAY:
        plt.axis('off')
        plt.show()
    return int(np.argmax(scores_gaus))

def get_pixel_size_um_czi(czi: CziFile):
    paths = [
        ".//Scaling/Items/Distance[@Id='X']/Value",
        ".//Distance[@Id='X']/Value",
    ]
    try:
        meta = czi.meta
        for path in paths:
            node = meta.find(path)
            if node is not None and node.text:
                val_m = float(node.text)
                return val_m * 1e6
    except Exception: pass # Still dk how to handle the exceptions
    try:
        meta = czi.meta_root
        for path in paths:
            node = meta.find(path)
            if node is not None and node.text:
                val_m = float(node.text)
                return val_m * 1e6 # to micron, seems like czi internal unit uses meters
    except Exception: pass
    print("Pixel size not found in metadata, default to None!")
    return None

def get_time_stamps_mins_czi(czi: CziFile, time_unit="min"):
    paths = [
        ".//Experiment/ExperimentBlocks/AcquisitionBlock/SubDimensionSetups/TimeSeriesSetup/Duration/TimeSpan/Value",
    ]
    try:
        meta = czi.meta_root
        for path in paths:
                node = meta.find(path)
                if node is not None and node.text:
                    val_s = float(node.text)
                    if time_unit == "min":
                        return val_s / 60.0 # in mins, seems like internal unit is secs
                    elif time_unit in ["s", "sec"]:
                        return val_s # in secs
                    elif time_unit in ["h", "hr", "hour"]:
                        return val_s / 3600.0 # in hours
                    else:
                        print(f"Unknown time unit: {time_unit}, default to mins, plz check!")
                        return val_s / 60.0 # default to mins
    except Exception: pass
    print("Time stamps not found in metadata, default to None!")
    return None

def save_np_tiff(path:str, img_np, pixel_size_um:float|None,axes_str:str, time_step:float|None, time_unit:str|None, z_picked:int):
    if not (path.endswith('.tiff') or path.endswith('.tif')): print("Save path not ends with \".tif\" or \".tiff\". Return.");return
    nomi = int((1/pixel_size_um) * 1e6)
    # print(f"tiff save img shape = {img_np.shape}; axes = {axes_str}")
    tifffile.imwrite(path, img_np,
                     shape=img_np.shape,
                     imagej=True,
                     dtype=img_np.dtype,
                     software='ImageJ',
                     resolution=((nomi, int(1e6)), (nomi, int(1e6))),
                     metadata={
                            'unit': 'micron',
                            'axes': axes_str,
                            'PhysicalSizeX': pixel_size_um,
                            'PhysicalSizeY': pixel_size_um,
                            'PhysicalSizeXUnit': 'um',
                            'PhysicalSizeY': 'um',
                            'Timestep': time_step,
                            'Timeunit':time_unit,
                            'Z_plane':z_picked,
                            'Description':f"Timestep:{time_step}, Z plane: {z_picked}"
                            })

### Methods for mp4 overlay
def min_to_hhmm(minutes: float) -> str:
    """Format time in Minute to HH:MM format"""
    total_min = int(round(minutes))
    hh = total_min // 60
    mm = total_min % 60
    return f"{hh:02d}:{mm:02d}"

def draw_overlay(frame_bgr: np.ndarray,
                 timestamp_str: str,
                 scalebar_px: int,
                 scalebar_um: float) -> np.ndarray:
    """Put scalebar + tag at right bottom; timestamp at left upper."""
    
    # if DEBUG: print(f"img shape = {frame_bgr.shape}")
    img = frame_bgr.copy()
    h, w = img.shape[:2]
    margin = 20  # How many pixel to the edge.

    # ── Scale bar ──────────────────────────────
    if scalebar_px and scalebar_px > 0:
        bar_x2 = w - margin
        bar_x1 = bar_x2 - scalebar_px
        bar_y  = h - margin
        bar_thick = max(3, h // 120)

        # Black Background (prevent white scalebar merge with white pixels)
        cv2.line(img, (bar_x1, bar_y), (bar_x2, bar_y), (0, 0, 0), bar_thick + 2)
        cv2.line(img, (bar_x1, bar_y), (bar_x2, bar_y), (255, 255, 255), bar_thick)

        # Scale bar tag
        label = f"{int(scalebar_um)} micron"
        (lw, lh), _ = cv2.getTextSize(label, FONT, FONT_SCALE * 0.85, FONT_THICK)
        lx = bar_x1 + (scalebar_px - lw) // 2
        ly = bar_y - bar_thick - 5
        cv2.putText(img, label, (lx, ly), FONT, FONT_SCALE * 0.85,
                    (0, 0, 0), FONT_THICK + 2, cv2.LINE_AA)
        cv2.putText(img, label, (lx, ly), FONT, FONT_SCALE * 0.85,
                    FONT_COLOR, FONT_THICK, cv2.LINE_AA)

    # ── Timestamp ──────────────────────────────
    (tw, th), _ = cv2.getTextSize(timestamp_str, FONT, FONT_SCALE, FONT_THICK)
    tx = margin + tw
    ty = margin + th
    
    # cv2.rectangle(img, (tx - pad, ty - th - pad), (tx + tw + pad, ty + pad),
    #               BG_COLOR, -1)
    cv2.putText(img, timestamp_str, (tx, ty), FONT, FONT_SCALE,
                FONT_COLOR, FONT_THICK, cv2.LINE_AA)
    # if DEBUG: print(f"Final img shape {img.shape}")
    return img

def main():
    if not (DEBUG or SAVE_NPY or SAVE_TIFF or SAVE_MP4):
        print("Save to neither tiff, npy, nor mp4. No need to run the code?")
        sys.exit()

    # Input folder paths
    parent_folder = None
    if DEBUG:   parent_folder = INPUT_FOLDER_PATH # r"~/test_C01"
    else:       parent_folder = get_filepath()
    print(f"parent_folder: {parent_folder}")
    p = Path(parent_folder).expanduser()
    if not p.exists(): print(f"The provided path does not exist: {p}"); sys.exit(1)
    if not p.is_dir(): print(f"The provided path is not a directory: {p}"); sys.exit(1)
    else:              print(f"Processing folder: {p}")
    
    # Find Czi Files
    # czi_files = sorted([f for f in p.glob('*.czi') if f.is_file()])
    czi_files = sorted(glob.glob(os.path.join(p, '*.czi')), key=lambda p: natural_sort_key(os.path.basename(p)))
    print(czi_files)
    if len(czi_files) == 0:
        print("try deeper directories")
        czi_files = sorted(glob.glob(os.path.join(p, '**', '*.czi')), key=lambda p: natural_sort_key(os.path.basename(p)))
        # czi_files = sorted([f for f in p.glob('**/*.czi') if f.is_file()])
    if len(czi_files) <= 0:
        print("No Czi Files Found. Exit()")
        sys.exit()
    else:
        print(f'Found {len(czi_files)} .czi files:')

    # If there are Czi Files.
    global OUTPUT_FOLDER, OUTPUT_TIFF, OUTPUT_NPY, OUTPUT_MP4
    if OUTPUT_FOLDER is None:OUTPUT_FOLDER = p / 'outputs'
    else: OUTPUT_FOLDER = Path(OUTPUT_FOLDER)
    '''
    if OUTPUT_FOLDER.exists():
        if not DEBUG:
            print(f"The file or directory already exists, exiting...")
            sys.exit(1)
        # if not in debug mode, exit to avoid overwriting existing files. In debug mode, we can ignore this error.
        # I am thinking should I check whether every files are processed and if not, only process the unprocessed files?
        # But for now, I will just exit.
    '''
    if SAVE_TIFF is not None and OUTPUT_TIFF is None: OUTPUT_TIFF = OUTPUT_FOLDER / 'tiff'
    if SAVE_NPY is not None and OUTPUT_NPY is None: OUTPUT_NPY = OUTPUT_FOLDER / 'npy'
    if SAVE_MP4 is not None and OUTPUT_MP4 is None: OUTPUT_MP4 = OUTPUT_FOLDER / 'mp4'

    # Loop through czis to pick best focus, normalize, and save
    global PIXEL_SIZE, TIME_STEP, TIME_UNIT
    for f in czi_files:
        print(f"Processing: {f}")
        czi = CziFile(str(f))
        dims = czi.get_dims_shape() # list of dictionary
        shape_dict = dims[0] if dims else {}
        frame = shape_dict.get('T', (0, 1))[1]   # time points
        z_planes = shape_dict.get('Z', (0, 1))[1]   # z-planes
        z_picked = best_focus_z(czi, frame-1, z_planes)
        # if DEBUG: print(shape_dict)
        img, _ = czi.read_image(Z=z_picked)
        # if DEBUG: print(f"shape of img: {img.shape}")
        squeezed_img = np.squeeze(img)
        # if DEBUG: print(f"shape of squeezed_img: {squeezed_img.shape}")
        norm_squeezed_img = np.stack([normalize_frame(frame) for frame in squeezed_img])
        if PIXEL_SIZE is None:
            PIXEL_SIZE = get_pixel_size_um_czi(czi)
        # if DEBUG: print(f"pixel_size {PIXEL_SIZE}")
        if TIME_STEP is None:
            duration = get_time_stamps_mins_czi(czi) # in mins, if exist in metadata
            TIME_STEP = 1 if duration is None else duration / (frame - 1) # in mins, if duration exist in metadata, otherwise assume 1 min per frame
        # if DEBUG: print(f"time_step {TIME_STEP} unit {TIME_UNIT}")
        f_stem = Path(f).stem
        if SAVE_TIFF:
            # print(dims)
            if not OUTPUT_TIFF.exists(): OUTPUT_TIFF.mkdir(parents=True)
            if dims: axes_str = [c for c in czi.dims if c!='Z' and shape_dict[c][1]-shape_dict[c][0]>1]
            else: axes_str = [""]
            save_np_tiff(path = str(OUTPUT_TIFF / (f_stem + '.tiff')),
                         img_np = norm_squeezed_img,
                         pixel_size_um = PIXEL_SIZE,
                         axes_str = "".join(axes_str),
                         time_step = TIME_STEP,
                         time_unit=TIME_UNIT,
                         z_picked = z_picked
                         )
        if SAVE_NPY:
            if not OUTPUT_NPY.exists(): OUTPUT_NPY.mkdir(parents=True)
            np.save(str(OUTPUT_NPY/(f_stem + '.npy')), norm_squeezed_img)
            metadata = {
                "pixel_size_um":    PIXEL_SIZE,
                "time_step":        TIME_STEP,
                "time_unit":        TIME_UNIT,
                "z_plane":          z_picked,
            }
            with open(str(OUTPUT_NPY/(f_stem+'.json')),'w',encoding='utf-8') as f:
                json.dump(metadata, f, indent=4)
        if SAVE_MP4:
            if not OUTPUT_MP4.exists(): OUTPUT_MP4.mkdir(parents=True)
            # if DEBUG: print(str(OUTPUT_MP4/(f_stem + '.mp4')))
            out = cv2.VideoWriter(str(OUTPUT_MP4/(f_stem + '.mp4')),
                                  cv2.VideoWriter_fourcc(*'mp4v'),
                                  OUTPUT_FPS,
                                  (norm_squeezed_img.shape[2], norm_squeezed_img.shape[1])
                                  )
            time_stamp = 0
            for slice in norm_squeezed_img:
                frame_bgr  = cv2.cvtColor(slice, cv2.COLOR_GRAY2BGR)

                # Timestamp minute to hhmm str
                ts_str = min_to_hhmm(time_stamp)
                frame_out = draw_overlay(frame_bgr, ts_str,
                                         round(SCALEBAR_UM/PIXEL_SIZE) if PIXEL_SIZE is not None else 0,
                                         SCALEBAR_UM)
                out.write(frame_out)
                time_stamp = (time_stamp + TIME_STEP) if TIME_STEP is not None else (time_stamp + 1)
            out.release()


if __name__ == "__main__":
    main()