###
# The purpose of this code is to:
# 1. Read time-lapse czi files, should be 3 layer z stack with time-lapse of difference time duration.
# 2. Pick the best focal plane.
# 3. Output to (can choose): (1) tiff. (2) npy (image data) + json (metadata). (3) mp4.
### Something to be aware of:
# 1. Two path (Input path, output path) needed, or will output besides the input path
# 2. All analysis shares the same time_step and pixel_to_um value. (to reduce times to reach metadata)


from typing import Final
### Parameters
# These are constant parameters for you to set as you need before run.
# The Final[] thing is just for type hinting, so that I will not accidentally change them during the code is running.
DEBUG: Final[bool] = False # If true, the path will take debug path (testing env)
IMG_DISPLAY: Final[bool] = False # If true, import mayplotlib and show all three z images to pick
# Input folder path input
INPUT_FOLDER_PATH: Final[str|None] = None # "/mnt/d/Osmolarity/2026-08-06/" # If not None, use this path as the input folder path, otherwise ask user to select a folder. Only used when DEBUG is False.
# Output folders path input
OUTPUT_FOLDER: str|None = None # "/mnt/d/Osmolarity/260812/mp4_resize/0.pre/"            # Will be changed if None
# These three will always be None
OUTPUT_TIFF: str|None = None              # output folder of tiff files, Will be changed if None
OUTPUT_NPY: str|None = None               # output folder of npy/json files, Will be changed if None
OUTPUT_MP4: str|None = None               # output folder of mp4 files, Will be changed if None
OUTPUT_MP4_RESIZE: str|None = None
# Metadata parameters (for movie overlay mostly)
PIXEL_SIZE: float|None = None
TIME_UNIT: Final[str] = "min"       # I guess we should always use min, because PIV analysis use min??? dk
PLOT_TIME_UNIT: Final[str] = "min"  # You can change this one, Time unit for plotting.
TIME_STEP: float|None = None
# Save Parameters
SAVE_TIFF: Final[bool] = False      # Whether you want tifffile
SAVE_NPY: Final[bool] = True       # Whether you want npy/json
SAVE_MP4: Final[bool] = False       # Whether you want mp4
SAVE_MP4_RESIZE: Final[bool] = True
### MP4 parameters
import cv2
DESIRED_HEIGHT = 540
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

import sys
from pathlib import Path
from aicspylibczi import CziFile
import numpy as np
if IMG_DISPLAY: import matplotlib.pyplot as plt
if SAVE_TIFF  : import tifffile
if SAVE_NPY   : import json

if __name__ == "__main__" and __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    __package__ = 'ImgProc'
from . import utils

''' # Get Filepath old code, try switch to utils
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
'''

''' # Natural_sort_key old code try switch to utils
def natural_sort_key(s):
    """sort filename (e.g. _2_ < _10_)"""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]
'''

''' # Normalize_frame old code try switch to utils
def normalize_frame(frame: np.ndarray) -> np.ndarray:
    # if frame.dtype == np.uint8: return frame
    f_min, f_max = np.percentile(frame, (1,99))
    # f_min, f_max = frame.min(), frame.max()
    if f_max == f_min:
        return np.zeros_like(frame, dtype=np.uint8)
    norm = (frame.astype(np.float32) - f_min) / (f_max - f_min) * 255
    return norm.astype(np.uint8)
'''

def best_focus_z(czi, t, z_planes):
    ''' Akash's matlab code for best focal plane selection:

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
        frame = utils.normalize_frame(np.squeeze(frame))
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
    """
    We want to get time-steps (in whatever time units) from czi file metadata.
    """
    paths = [
        ".//Experiment/ExperimentBlocks/AcquisitionBlock/SubDimensionSetups/TimeSeriesSetup/Interval/TimeSpan/Value",
    ] # This should get time_steps in sec
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
    if not (DEBUG or SAVE_NPY or SAVE_TIFF or SAVE_MP4 or SAVE_MP4_RESIZE):
        print("Save to neither tiff, npy, nor mp4. No need to run the code?")
        sys.exit()

    # Input folder paths
    parent_folder = None
    if DEBUG:   parent_folder = INPUT_FOLDER_PATH # r"~/test_C01"
    else:       parent_folder = utils.get_filepath(sys_argv=sys.argv,given_path=INPUT_FOLDER_PATH) # parent_folder = get_filepath()
    # I want to change this bc this "sys.argv" might be dangerous?

    print(f"parent_folder: {parent_folder}")
    if type(parent_folder) != str or parent_folder == "": print(f"The provided path does not exist: {parent_folder}"); sys.exit(1)
    path_parent_folder = Path(parent_folder).expanduser()
    if not path_parent_folder.exists(): print(f"The provided path does not exist: {path_parent_folder}"); sys.exit(1)
    if not path_parent_folder.is_dir(): print(f"The provided path is not a directory: {path_parent_folder}"); sys.exit(1)
    else:                               print(f"Processing folder: {path_parent_folder}")
    
    # Find Czi Files
    # czi_files = sorted([f for f in path_parent_folder.glob('*.czi') if f.is_file()])
    czi_files = sorted(path_parent_folder.rglob('*.czi'), key=lambda p: utils.natural_sort_key(p.name))
    for f in czi_files:
        print(utils.remove_date_folder(f.relative_to(path_parent_folder)))
    

    ''' # I think we dont need this part if I use rglob, bc already recursively search
    if len(czi_files) == 0:
        print("try deeper directories")
        czi_files = sorted(glob.glob(os.path.join(p, '**', '*.czi')), key=lambda p: utils.natural_sort_key(os.path.basename(p)))
        # czi_files = sorted([f for f in p.glob('**/*.czi') if f.is_file()])
    '''

    if len(czi_files) <= 0:
        print("No Czi Files Found. Exit()")
        sys.exit()
    else:
        print(f'Found {len(czi_files)} .czi files.')

    # If there are Czi Files, determining output folders.
    global OUTPUT_FOLDER
    if OUTPUT_FOLDER is None: OUTPUT_FOLDER = utils.get_filepath(sys_argv = sys.argv[1:])
    if OUTPUT_FOLDER is None or OUTPUT_FOLDER == '':OUTPUT_FOLDER = path_parent_folder / 'outputs'
    else: OUTPUT_FOLDER = Path(OUTPUT_FOLDER)
    print(f"OUTPUT_FOLDER: {OUTPUT_FOLDER}")

    '''
    if OUTPUT_FOLDER.exists():
        if not DEBUG:
            print(f"The file or directory already exists, exiting...")
            sys.exit(1)
        # if not in debug mode, exit to avoid overwriting existing files. In debug mode, we can ignore this error.
        # I am thinking should I check whether every files are processed and if not, only process the unprocessed files?
        # But for now, I will just exit.
    '''
    

    # for f in czi_files:
    #     print((OUTPUT_MP4/utils.remove_date_folder(f.relative_to(path_parent_folder))))

    # Loop through czis to pick best focus, normalize, and save
    global OUTPUT_TIFF, OUTPUT_NPY, OUTPUT_MP4, OUTPUT_MP4_RESIZE
    global PIXEL_SIZE, TIME_STEP, TIME_UNIT
    for f in czi_files:
        print(f"Processing: {f}")

        # Determin whether this file has been processed (NOT YET FINISHED, HAVE LOGICAL BUG!)
        f_stem = Path(f).stem
        do_processing = False
        # Now everytime OUTPUT_TIFF, OUTPUT_NPY, OUTPUT_MP4 will be updated everytime
        if SAVE_TIFF: OUTPUT_TIFF = (OUTPUT_FOLDER / 'tiff'/ utils.remove_date_folder(f.relative_to(path_parent_folder))).parent
        if SAVE_NPY : OUTPUT_NPY  = (OUTPUT_FOLDER / 'npy' / utils.remove_date_folder(f.relative_to(path_parent_folder))).parent
        if SAVE_MP4 : OUTPUT_MP4  = (OUTPUT_FOLDER / 'mp4' / utils.remove_date_folder(f.relative_to(path_parent_folder))).parent
        if SAVE_MP4_RESIZE: OUTPUT_MP4_RESIZE  = (OUTPUT_FOLDER / 'mp4_resize' / utils.remove_date_folder(f.relative_to(path_parent_folder))).parent
        norm_squeezed_img = None
        if SAVE_MP4_RESIZE  and not (OUTPUT_MP4_RESIZE  / f"{f_stem}.mp4" ).exists(): do_processing = True
        if SAVE_MP4  and not (OUTPUT_MP4_RESIZE  / f"{f_stem}.mp4" ).exists(): do_processing = True
        if SAVE_TIFF and not (OUTPUT_TIFF / f"{f_stem}.tiff").exists(): do_processing = True
        if SAVE_NPY:
            if (OUTPUT_NPY / f"{f_stem}.npy" ).exists() and \
               (OUTPUT_NPY / f"{f_stem}.json").exists():
                do_processing = False
                norm_squeezed_img = np.load(str(OUTPUT_NPY / f"{f_stem}.npy"))
                if (PIXEL_SIZE is None) or (TIME_STEP is None) or (TIME_UNIT is None):
                    meta = None
                    try:
                        with open(str(OUTPUT_NPY / f"{f_stem}.json"),'r') as j:
                            meta = json.load(j)
                    except Exception as e:
                        print(f"Failed to open or load json file: {str(OUTPUT_NPY / f'{f_stem}.json')}. Error: {e}")
                    if meta is not None:
                        if PIXEL_SIZE is None: PIXEL_SIZE = meta["pixel_size_um"]
                        if TIME_STEP is None: TIME_STEP = meta["time_step"]
                        if TIME_UNIT is None: TIME_STEP = meta["time_unit"]
            else: do_processing = True
            
        if do_processing:
            czi = CziFile(str(f))
            if all([i == 0 for i in czi.size]): print(f"All dimentions is 0."); continue
            dims = czi.get_dims_shape() # list of dictionary
            shape_dict = dims[0] if dims else {}
            frame = shape_dict.get('T', (0, 1))[1]   # time points
            z_planes = shape_dict.get('Z', (0, 1))[1]   # z-planes
            z_picked = best_focus_z(czi, frame-1, z_planes)
            # if DEBUG: print(shape_dict)
            img, _ = czi.read_image(Z=z_picked)
            # if DEBUG: 
            print(f"shape of img: {img.shape}")
            squeezed_img = np.squeeze(img)
            if len(squeezed_img.shape) < 3: print(f"Dimention {squeezed_img.shape} less than 3. next file"); continue
            # if DEBUG: print(f"shape of squeezed_img: {squeezed_img.shape}")
            norm_squeezed_img = np.stack([utils.normalize_frame(frame) for frame in squeezed_img])
            if PIXEL_SIZE is None:
                PIXEL_SIZE = get_pixel_size_um_czi(czi)
            # if DEBUG: print(f"pixel_size {PIXEL_SIZE}")
            if TIME_STEP is None:
                TIME_STEP = get_time_stamps_mins_czi(czi) # in mins, if exist in metadata
                if TIME_STEP is None: TIME_STEP = 1 # in mins, if duration exist in metadata, otherwise assume 1 min per frame
                print(f"time_step {TIME_STEP}, unit {TIME_UNIT}")
            # if DEBUG: print(f"time_step {TIME_STEP} unit {TIME_UNIT}")
        
        if SAVE_TIFF and not (OUTPUT_TIFF / (f_stem + '.tiff')).exists():
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
        if SAVE_NPY  and not (OUTPUT_NPY  / (f_stem + '.npy' )).exists():
            if not OUTPUT_NPY.exists(): OUTPUT_NPY.mkdir(parents=True)
            np.save(str(OUTPUT_NPY / (f_stem + '.npy')), norm_squeezed_img)
            metadata = {
                "pixel_size_um":    PIXEL_SIZE,
                "time_step":        TIME_STEP,
                "time_unit":        TIME_UNIT,
                "z_plane":          z_picked,
            }
            with open(str(OUTPUT_NPY / (f_stem+'.json')),'w',encoding='utf-8') as f:
                json.dump(metadata, f, indent=4)
        if norm_squeezed_img is not None: norm_squeezed_img = (norm_squeezed_img / 255 * 220 + 35).astype(np.uint8)
        if SAVE_MP4  and not (OUTPUT_MP4  / (f_stem + '.mp4' )).exists():
            if not OUTPUT_MP4.exists(): OUTPUT_MP4.mkdir(parents=True)
            # if DEBUG: print(str(OUTPUT_MP4/(f_stem + '.mp4')))
            out = cv2.VideoWriter(str(OUTPUT_MP4 / (f_stem + '.mp4')),
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
        if SAVE_MP4_RESIZE  and not (OUTPUT_MP4_RESIZE  / (f_stem + '.mp4' )).exists():
            if not OUTPUT_MP4_RESIZE.exists(): OUTPUT_MP4_RESIZE.mkdir(parents=True)
            # if DEBUG: print(str(OUTPUT_MP4/(f_stem + '.mp4')))
            out = None
            time_stamp = 0
            scale_factor = DESIRED_HEIGHT / norm_squeezed_img.shape[1]
            for slice in norm_squeezed_img:
                frame_bgr  = cv2.cvtColor(slice, cv2.COLOR_GRAY2BGR)
                # Timestamp minute to hhmm str
                ts_str = min_to_hhmm(time_stamp)
                frame_out = draw_overlay(frame_bgr, ts_str,
                                         round(SCALEBAR_UM/PIXEL_SIZE) if PIXEL_SIZE is not None else 0,
                                         SCALEBAR_UM)
                frame_resize = cv2.resize(frame_out,dsize=None,fx=scale_factor,fy=scale_factor,interpolation=cv2.INTER_LINEAR)
                if out is None:
                    out = cv2.VideoWriter(str(OUTPUT_MP4_RESIZE / (f_stem + '.mp4')),
                                          cv2.VideoWriter_fourcc(*'mp4v'),
                                          OUTPUT_FPS,
                                          (frame_resize.shape[1], frame_resize.shape[0])
                                         )
                out.write(frame_resize)
                time_stamp = (time_stamp + TIME_STEP) if TIME_STEP is not None else (time_stamp + 1)
            out.release()
if __name__ == "__main__":
    main()