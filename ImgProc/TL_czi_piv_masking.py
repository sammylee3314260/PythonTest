###
# The purpose of this code is to:
# 1. Read in image files.
# 2. Do some image pre-processing.
# 3. Run PIV test, save PIV result.
# 4. Run chan-vese masking, save mask.
# 5. Quantify masked PIV result, save result.

from typing import Final, List
### Some Parameters
DEBUG: Final[bool] = False
INPUT_PATH: Final[str|None|List[str]] = None  # You can put file path here, or None for other input options
FILE_TYPE: str|None = None          # 'npy' or 'tiff' or None (auto determine)
PIXEL_SIZE: float|None = None
TIME_STEP: float|None = 15.0        # Time steps in min
TIME_UNIT: str|None = "min"          # Time Unit
MULTIPROCESS: Final[bool] = False
WORKERS_LIMIT: Final[int] = 2
SAVE_PIV_VIDEOS:Final[bool] = False # Whether to save piv results to mp4
OUTPUT_FPS: Final[int] = 10
OUTPUT_FOLDER: str|None = None
TEST_GAUSSIAN:Final[bool] = True    # If true use gaussian window passes for variance masking, if false use sobel
# PIV parameters
# window size = [128, 64, 32, 16]. Passes: 3.

import openpiv.settings as settings
def piv_parameter_init(time_step: float = TIME_STEP, is_debug: bool = DEBUG):
    return settings.PIVSettings(
        windowsizes=[128, 64, 32, 16],
        overlap=[64, 32, 16, 8],
        num_iterations=3,
        correlation_method="circular",       # Default
        interpolation_order=1,               # I guess this means linear in matlab PIVlab
        deformation_method="second image",   # I think matlab PIVlab only deforms the second img
        roi="full",                          # Default:analyse the full image
        dt=time_step * 60.0,         # time interval between frames, in sec
        show_all_plots=False and is_debug    # Dont show plot, plot urself or the prog will stall
        )
### End of Parameters

from pathlib import Path
import sys
import glob
import os
import tifffile
import json
import numpy as np
import openpiv.windef as windef
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import gc
# import matplotlib.pyplot as plt
import cv2
from skimage import exposure, filters, segmentation, measure, transform
from scipy import ndimage
import scipy.signal
import pandas as pd

if __name__ == "__main__" and __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    __package__ = 'ImgProc'
from . import utils

def run_simple_multipass(frame_pairs):
    global TIME_STEP, DEBUG
    piv_parameter = piv_parameter_init(TIME_STEP,DEBUG)
    try: return windef.simple_multipass(frame_pairs[0], frame_pairs[1], settings=piv_parameter)
    except ValueError as e: printf(f"ValueError {e}"); return None

def save_piv_videos_cv2(img, x,y,u,v, output_path,step=1, scale=1.0,thickness=1, color=(0,255,0)):
    # This is a placeholder function to save PIV result as video using cv2. You can implement it based on your needs.
    writer = cv2.VideoWriter(str(output_path),
                             cv2.VideoWriter_fourcc(*'mp4v'),
                             OUTPUT_FPS,
                             (img.shape[2], img.shape[1])
                             )
    
    for t in range(img.shape[0]-1):
        frame_bgr = cv2.cvtColor(img[t], cv2.COLOR_GRAY2BGR)
        # Overlay timestamp and scale bar
        # timestamp_str = f"T={t*TIME_STEP} min"
        # scalebar_px = int(SCALEBAR_UM / pixel_size_um) if pixel_size_um else 0
        # frame_with_overlay = draw_overlay(frame_bgr, timestamp_str, scalebar_px, pixel_size_um, SCALEBAR_UM)
        # Overlay PIV vectors

        iterable = zip(x[t,::step,::step].flatten(), y[t,::step,::step].flatten(), u[t,::step,::step].flatten(), v[t,::step,::step].flatten())
        for (x_i, y_i, u_i, v_i) in iterable:
            start_point = (int(x_i), int(y_i))
            end_point = (int(x_i + scale * u_i), int(y_i + scale * v_i))
            cv2.arrowedLine(frame_bgr, start_point, end_point, color, thickness, cv2.LINE_AA, tipLength=0.5)
        writer.write(frame_bgr)
    writer.release()

def sobel_gradient(img):
    # sobel_gradient is the find_Edge from Akash's matlab code, without the manual border manipulation.
    img = np.asarray(img, dtype=np.float64)
    kernal_v = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    kernal_h = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    edge_v = scipy.signal.convolve2d(img, kernal_v, mode='same', boundary='symm')
    edge_h = scipy.signal.convolve2d(img, kernal_h, mode='same', boundary='symm')
    C = np.hypot(edge_v, edge_h)
    return C

def get_gaussian_gradient(img):
    sigmas = (2, 4, 8, 16)
    grads = []
    for s in sigmas:
        smoothed = ndimage.gaussian_filter(img, sigma=s)
        gx = filters.sobel_h(smoothed)
        gy = filters.sobel_v(smoothed)
        g = np.hypot(gx, gy)
        grads.append(g)
    return np.max(np.stack(grads, axis=0), axis=0)

def image_pre_process(img):
    # img = io.imread(str(path)).astype('float32') / 65535.0
    img = filters.unsharp_mask(img)
    img = exposure.equalize_adapthist(img)
    if TEST_GAUSSIAN: img = get_gaussian_gradient(img)
    else: img = sobel_gradient(img)
    return img

def run_chan_vese(img, is_tqdm = True):
    thres = filters.threshold_otsu(img)
    init = img > thres
    # Because defaule tol = 0.001. /
    # if you dont want tol I think I should set it to 0.
    seg, _, eng = segmentation.chan_vese(img, mu=0.2, max_num_iter=100, tol = 0, extended_output=True, lambda1=0.97, lambda2=1, init_level_set=init)
    # seg, _, eng = segmentation.chan_vese(img, mu=0.2, max_num_iter=100,tol=1e-3,extended_output=True)
    # Remember if end = "", stdout buffer won't flush automatically!!
    # so no printing before tqdm
    if is_tqdm: tqdm.write(f" {len(eng)}",end="")
    else:       print(f" {len(eng)}",end="")
    return seg

def get_largest_region(mask_2d):
    labelled = measure.label(mask_2d)
    props = measure.regionprops(labelled)
    if not props:# If all mask is false
        return np.zeros_like(mask_2d)
    largest = max(props, key = lambda r: r.area)
    return ndimage.binary_fill_holes(labelled == largest.label)

def main():
    ### Step 1: As always, readin paths
    parent_folder = ""
    global FILE_TYPE
    if DEBUG:
        if FILE_TYPE == 'npy': parent_folder = r"~/test_C01/outputs/npy"
        elif FILE_TYPE == 'tiff': parent_folder = r"~/test_C01/outputs/tiff"
        else: parent_folder = r"~/test_C01/outputs/npy"
    else:     parent_folder = utils.get_filepath(try_gui=False, sys_argv=sys.argv, given_path=INPUT_PATH)
    path = Path(parent_folder).expanduser()
    if not path.exists(): print(f"The provided path does not exist: {path}"); sys.exit(1)
    if not path.is_dir(): print(f"The provided path is not a directory: {path}"); sys.exit(1)
    else:                 print(f"Processing folder: {path}")

    # Notice 'files' is a str list not a pathlib POXIS object list!!!
    files = [p for p in glob.glob(os.path.join(path, '*')) if Path(p).expanduser().is_file()]
    if len(files) == 0: files = [p for p in glob.glob(os.path.join(path, '**', '*')) if Path(p).expanduser().is_file()]
    if len(files) == 0: print(f"Cannot find files in and under dir {INPUT_PATH}, exit."); sys.exit()
    files = sorted(files, key=lambda p: utils.natural_sort_key(os.path.basename(p)))
    
    # Check whether there are tiff files or npy/json files. This is an implementation so that this script can process tiff or npy/json files
    # I want to change files from str to poxis object?
    files_pathlib = [Path(f) for f in files]
    print(f"files:{files}")
    if FILE_TYPE is None:
        extensions = [f.suffix for f in files_pathlib]
        if '.npy' in extensions: FILE_TYPE = 'npy'
        elif '.tiff' in extensions or '.tif' in extensions: FILE_TYPE = 'tiff'
        if DEBUG: print(f"Initial FILE_TYPE is None, now {FILE_TYPE}")
    if FILE_TYPE == 'npy':
        # How to check whether every npy file is with a json file?
        npys = [f for f in files_pathlib if f.suffix == '.npy']
        not_processed = []
        processed = []
        for f in npys:
            if DEBUG:
                print(os.path.join(str(f.parent),f"{str(f.stem)}.json"))
            if (os.path.join(str(f.parent),f"{str(f.stem)}.json")) not in files: not_processed.append(f)
            else: processed.append(f)
        if len(not_processed) != 0:
            print(f"There are some npy without paired json files, please check.\n")
            for i in not_processed: print(str(i))
        if len(processed) == 0: print(f"There is no npy files paired with json files, exit()"); sys.exit()
    elif FILE_TYPE == 'tiff':
        processed = [f for f in files_pathlib if f.suffix in ['.tif', '.tiff']]
        if len(processed) == 0: print(f"There is no tiff files, exit()"); sys.exit()
        pass
    else: print(f"File type {FILE_TYPE} not defined? exit"); sys.exit()

    global PIXEL_SIZE, TIME_STEP, TIME_UNIT, OUTPUT_FOLDER
    OUTPUT_FOLDER = path / 'analysis_output'
    if OUTPUT_FOLDER.exists():
        if not DEBUG: print(f"output folder {OUTPUT_FOLDER} exists, exit?"); sys.exit()
    else: OUTPUT_FOLDER.mkdir() # No need to set parent bc 'path' must exist.
    
    # get cpu for parallel processing
    get_cpu = os.environ.get('SLURM_CPUS_PER_TASK') or \
              os.environ.get('SLURM_CPUS_ON_NODE') or \
              os.environ.get('SLURM_JOB_CPUS_PER_NODE')
    if get_cpu is None:
        iscluster =  False
        workers = os.cpu_count()
        print(f"Not on SLURM clusters, cpu cores = {workers}")
    else:
        # if on cluster, the cpu_count return value will be all CPU numbers on the cluster
        # which is very large, while the available CPU for you is limited
        iscluster = True
        workers = int(get_cpu)
        print(f"On SLURM clusters, cpu cores = {workers}")

    ### Step 2: start processing individual files
    # f or processed is poxis object list
    speed_dict = {}
    order_parameter_dict = {}
    for f in processed:
        print(f"file:{f}")
        ### Read files into numpy array and get pixel size and time steps.
        img = None
        # One problem is That after the three metadata parameters are determined, they are fixed,
        # so every file in this looping should have the same metadata parameters
        if FILE_TYPE == 'npy':
            img = np.load(str(f))
            if PIXEL_SIZE is None or TIME_STEP is None or TIME_UNIT is None:
                with open(str(f.parent / f"{f.stem}.json"),'r') as j:
                    meta = json.load(j)
                if PIXEL_SIZE is None: PIXEL_SIZE = meta["pixel_size_um"]
                if TIME_STEP is None: TIME_STEP = meta["time_step"]
                if TIME_UNIT is None: TIME_UNIT = meta["time_unit"]
                # z_picked = meta["z_plane"]
        elif FILE_TYPE == 'tiff':
            tiff = tifffile.TiffFile(str(f))
            img = tiff.asarray()
            if PIXEL_SIZE is None or TIME_STEP is None or TIME_UNIT is None:
                meta = tiff.imagej_metadata
                if PIXEL_SIZE is None: PIXEL_SIZE = meta["physicalsizex"]
                if TIME_STEP is None: TIME_STEP = meta["timestep"]
                if TIME_UNIT is None: TIME_UNIT = meta["timeunit"]
                '''{
                'ImageJ': '1.11a', 'images': 13, 'frames': 13, 'hyperstack': True, 'mode': 'grayscale',
                'loop': False, 'unit': 'micron', 'physicalsizex': 0.645, 'physicalsizey': 'um', 'physicalsizexunit': 'um',
                'timestep': 15.0, 'timeunit': 'min', 'z_plane': 2, 'description': 'Timestep:15.0, Z plane: 2'
                }'''
        # Notice after the processing The three metadata might be None
        if PIXEL_SIZE is None: print(f"pixel size in um: {PIXEL_SIZE} is None. PIV process with unit in pixel!!"); PIXEL_SIZE = 1
        if TIME_STEP is None: print(f"pixel size in um: {TIME_STEP} is None. PIV process with unit in frame!!");TIME_STEP = 1
        if TIME_UNIT is None: print(f"no time unit found, you should take care of this yourself!!"); TIME_UNIT = ""

        ### Start PIV analysis
        frame = img.shape[0]
        norm_img = [utils.normalize_frame(f) for f in img]
        results = []
        if not MULTIPROCESS:
            for t in tqdm(range(frame-1), desc=f"Processing {f.stem}"):
                results.append(
                    run_simple_multipass((norm_img[t], norm_img[t+1]))
                    )
        else:
            image_pairs = [(norm_img[t], norm_img[t+1]) for t in range(frame-1)]
            with ProcessPoolExecutor(max_workers=min(workers, WORKERS_LIMIT)) as executor:
                results = list(tqdm(
                    executor.map(run_simple_multipass, image_pairs),
                    total=len(image_pairs),
                    desc = 'Multiprocess PIV processing'
                ))
            del image_pairs
            gc.collect() # force garbage collection after multiprocessing to free memory
        if any(r is None for r in results):
            print('There are None (should be error) in the PIV process, continue')
            continue
        x_result = np.stack([r[0] for r in results])
        y_result = np.stack([r[1] for r in results])
        u_result = np.stack([r[2] for r in results])
        v_result = np.stack([r[3] for r in results])
        speed = np.hypot(u_result, v_result)
        '''
        # If you want to plot the quiver plot of the PIV result, you can do it like this:
        plt.imshow(np.flip(norm_img[0],axis=0), cmap='gray') # plot the first frame
        plt.quiver(x_result[0], y_result[0], u_result[0], v_result[0], color='g') # plot arrow
        breakpoint() # check the quiver plot, you can save it or show it. If you show it, the program will stall until you close the plot.
        # plt.savefig(OUTPUT_FOLDER / 'quiver.png') # Can directly save this quiver/image overlay.
        '''
        if SAVE_PIV_VIDEOS:
            save_piv_videos_cv2(np.flip(norm_img, axis=1), x_result, y_result, u_result, v_result,
                                str(OUTPUT_FOLDER / f"{f.stem}_piv_result.mp4"),
                                scale=0.8,thickness=1)
        np.savez(str(OUTPUT_FOLDER / f"{f.stem}_piv_result.npz"), x=x_result, y=y_result, u=u_result, v=v_result)

        ### Run Chan-Vese
        img_proc = [image_pre_process(i) for i in norm_img[1:]] # No need to deal with first frame
        if not iscluster: 
            # if not on cluster just use single processor not use parallel,
            # seems like it becomes very slow on my computer?
            if DEBUG: print(f"shape of images = {img_proc.shape}")
            # del img_proc[0] # No need to deal with the first img

            results = []
            for i in tqdm(img_proc,  desc = 'Chan-Vese Processing'):
                results.append(run_chan_vese(i))
                # breakpoint() # check the results in cli mode exit()
            print("") # 1: put a \n, 2: flush stdout cache
        else:
            with ProcessPoolExecutor(max_workers=min(workers,WORKERS_LIMIT)) as executor:
                results = list(tqdm(
                    executor.map(run_chan_vese, img_proc),
                    total=len(img_proc),
                    desc = 'Chan-Vese Processing'
                ))
            print("") # 1: put a \n, 2: flush cache

        ### Post masking data analysis
        results = np.stack(results)
        results_new = np.stack([get_largest_region(mask) for mask in results])
        results_resize = np.stack([transform.resize(mask, speed[0].shape, order=0, preserve_range=True).astype(bool) for mask in results_new])

        average_speed = [np.nanmean(speed[i][results_resize[i]]) for i in range(frame-1)]
        principle_dir = [(np.nanmean(u_result[i][results_resize[i]]),\
                          np.nanmean(v_result[i][results_resize[i]]))\
                          for i in range(frame-1)]

        # order parameter: (u,v)\dot (prin_u,prin_v) / (|u,v|*|prin_u,prin_v|)
        order_parameter = [np.nanmean((u_result[i][results_resize[i]]*principle_dir[i][0] 
                                     + v_result[i][results_resize[i]]*principle_dir[i][1])
                                     /(speed[i][results_resize[i]] * np.hypot(*principle_dir[i])))
                                       for i in range(frame-1)]
        speed_dict[f.stem] = average_speed
        order_parameter_dict[f.stem] = order_parameter

        # Save masks (ideally w/ jpeg or mp4 for validation) and analysis results
        # Data Processing: 5. Save the results in csv files for further analysis.
    speed_df = pd.DataFrame(speed_dict)
    order_df = pd.DataFrame(order_parameter_dict)
    speed_df.index.name = 'Frame_index'
    order_df.index.name = 'Frame_index'

    targets = ['pre', 'exp', 'recov']
    output_pref = path.stem if any(t in path.stem for t in targets)\
        else path.parent.stem if any(t in path.parent.stem for t in targets)\
        else ''
    speed_df.to_csv(path / (str(output_pref) + 'average_speed.csv'))
    order_df.to_csv(path / (str(output_pref) + 'order_parameter.csv'))

if __name__ == "__main__":
    main()