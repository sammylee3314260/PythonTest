'''
This is a test transfer from Akash's matlab code 
"Folder_Mat_based_edge_cluster_version"
The goal is to use Chan-Vase algorithm to generate image mask
using brightness of area.
'''
isdebug = False
istestmask = True

# initial imports
import os
# Determine GUI for tkinter
def is_gui():
    if 'DISPLAY' in os.environ.keys():return os.environ['DISPLAY'] is not None
    else: return False
if is_gui():
    import tkinter as tk
    from tkinter import filedialog
# import readline
# Find all data (s* folders)
from pathlib import Path
# Imports for Data processing: 1
# mat file opener, img opener, img pre_processing, Chan-Vese, parallel processing
from scipy import ndimage
import scipy.io, scipy.signal
import numpy as np
from skimage import exposure, filters, segmentation, measure, transform
import skimage.io
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm
import matplotlib.pyplot as plt
# Imports for Data Processing: 5
import pandas as pd

# tkinter or CLI readline
def read_path():
    if is_gui():
        try:
            root = tk.Tk()
            root.withdraw()
            path = filedialog.askdirectory()
            root.destroy()
            print(f'Input folder: {path}')
            return path
        except tk.TckError:
            print("GUI is not available. Falling back to command line input.")
            pass
    path = input("Please enter the path to the parent folder: ")
    
    print(f'Input folder: {path}')
    return path

def sobel_gradient(img):
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

def load_pre_process(path):
    img = skimage.io.imread(str(path))
    # img = io.imread(str(path)).astype('float32') / 65535.0
    img = filters.unsharp_mask(img)
    img = exposure.equalize_adapthist(img)
    if istestmask: img = get_gaussian_gradient(img)
    else: img = sobel_gradient(img)
    return img


def run_chan_vese(img, is_tqdm = True):
    thres = filters.threshold_otsu(img)
    init = img > thres
    seg, _, eng = segmentation.chan_vese(img, mu=0.2, max_num_iter=100,extended_output=True, lambda1=0.97, lambda2=1, init_level_set=init)
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

if __name__ == "__main__":
    # File input

    if isdebug:
        parent_folder = '~/test_C01'
        print(f'Input folder: {parent_folder}')
    else:
        parent_folder = read_path()
    print(f"Input folder: {parent_folder}")
    print(type(parent_folder))
    if type(parent_folder) != str or not os.path.isdir(parent_folder):
        print("Invalid folder path. Please check and try again.")
        exit()
    '''
    design a argparse here? for further useage?
    '''

    # Find all data (s* folders)
    p = Path(parent_folder).expanduser()
    # print([d.name for d in p.iterdir()])
    # Try different possible subfolder structures, e.g. c1/s1, s1, etc.
    s_dirs = sorted([d for d in p.glob('c*/s*') if d.is_dir()])
    if len(s_dirs) == 0: s_dirs = sorted([d for d in p.glob('s*') if d.is_dir()])

    # get CPU core numbers for possible parallel processing
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

    speed_dict = {}
    order_parameter_dict = {}

    # Iteration through all s* subfolders.
    for d in s_dirs:
        # Data processing: 1. Open PIV lab mat file and calculate speed from u_smooth and v_smooth
        print(f"Processing folder: {d.name}")
        folderpath = d.absolute() # print(folderpath)
        PIV_file = list(folderpath.glob('z*/*mat'))
        if len(PIV_file)!=1:
            print(f"there are {len(PIV_file)} PIV Files:{PIV_file}")
            continue
        print(f"mat file name = {PIV_file[0].name}")
        data = scipy.io.loadmat(str(PIV_file[0]))
        try: frames = data['PIVresult'].item()
        except ValueError:
            print(f"file {PIV_file[0]} have {data['PIVresult'].shape} results?")
            frames = data['PIVresult'][0]
        # print(f"data['PIVresult'] = {data['PIVresult']}, frames = {frames}")
        '''
        >>> print(list(data.keys()))
        ['__header__', '__version__', '__globals__', 'PIVresult', 'S', 'a',
        'amount', 'b', 'correlation_map', 'direc', 'directory', 'filenames',
        'i', 'j', 'nr_of_cores', 'p', 'pairwise', 'parentDir', 'r', 'resultFile',
        's', 'sFolderPath', 'sFolders', 'slicedfilename1', 'slicedfilename2',
        'suffix', 'tifFiles', 'typevector', 'typevector_filt', 'u', 'u_filt',
        'u_smoothed', 'v', 'v_filt', 'v_smoothed', 'x', 'y', 'zFolder', 'zFolderPath']
        '''
        # np.hypot() for calculate sqrt(u^2 + v^2)
        u_smoothed = [data['u_smoothed'][i][0] for i in range(frames)]
        v_smoothed = [data['v_smoothed'][i][0] for i in range(frames)]
        speed = np.hypot(u_smoothed,v_smoothed) # print(f"shape of speed = {speed.shape}")
        print(f"shape of u_smoothed = {u_smoothed[0].shape}")

        # Data processing: 2. Tiff files analysis, mask generation.
        tif_files = sorted([d for d in folderpath.glob('z*/*.tif') if d.is_file()])
        # print(f"delete tif_files[0] = {str(tif_files[0])}")
        del tif_files[0] # No need to deal with the first frame tif file.
        if len(tif_files) != frames:
            print(f"frames = {frames}, but len(tif_files)) = {len(tif_files)}")
            continue
        if not iscluster: 
            # if not on cluster just use single processor not use parallel,
            # because too much context switching. TOO SLOW
            images = np.stack([load_pre_process(f) for f in tif_files])
            # breakpoint() # check the pre-processed images in cli mode
            # sobel_gradient is the find_Edge from Akash's matlab code, without the manual border manipulation.
            print(f"shape of images = {images[0].shape}")
            
            results = []
            for img in tqdm(images,  desc = 'Chan-Vese Processing'):
                results.append(run_chan_vese(img))
                # breakpoint() # check the results in cli mode exit()
            print("") # 1: put a \n, 2: flush stdout cache
        else:
            with ThreadPoolExecutor(max_workers=min(4, workers)) as executor:
                images = list(executor.map(load_pre_process, tif_files))
            with ProcessPoolExecutor(max_workers=workers) as executor:
                results = list(tqdm(
                    executor.map(run_chan_vese, images),
                    total=len(images),
                    desc = 'Chan-Vese Processing'
                ))
            print("") # 1: put a \n, 2: flush cache
        results = np.stack(results)
        results_new = np.stack([get_largest_region(mask) for mask in results])
        results_resize = np.stack([transform.resize(mask, speed[0].shape, order=0, preserve_range=True).astype(bool) for mask in results_new])
        # Data Processing: 2.5. Display the mask and original image for validation.

        # Data Processing: 3. Determine migration direction / Separate leader/follower cells
        # I think we can try some more direct way to survey leader/follower cells
        # But Akash use normal directions (UDLR) to analyse this
        # Data Processing: 4. Use the mask to calculate average PIV speed, order parameter and?
        # Ideally for leader and follower too, but i dont have masks right now.
        average_speed = [np.nanmean(speed[i][results_resize[i]]) for i in range(frames)]
        principle_dir = [(np.nanmean(u_smoothed[i][results_resize[i]]),\
                        np.nanmean(v_smoothed[i][results_resize[i]]))\
                        for i in range(frames)]
        # order parameter: (u,v)\dot (prin_u,prin_v) / (|u,v|*|prin_u,prin_v|)
        order_parameter = [np.nanmean((u_smoothed[i][results_resize[i]]*principle_dir[i][0] \
                                    + v_smoothed[i][results_resize[i]]*principle_dir[i][1])\
                                    / (speed[i][results_resize[i]] * np.hypot(*principle_dir[i])))\
                                    for i in range(frames)]
        speed_dict[d.name] = average_speed  
        order_parameter_dict[d.name] = order_parameter
        # if isdebug:breakpoint() # check the results in cli mode

    # Data Processing: 5. Save the results in csv files for further analysis.
    speed_df = pd.DataFrame(speed_dict)
    order_df = pd.DataFrame(order_parameter_dict)
    speed_df.index.name = 'Frame_index'
    order_df.index.name = 'Frame_index'

    targets = ['pre', 'exp', 'recov']
    output_pref = p.name if any(t in p.name for t in targets)\
        else p.parent.name if any(t in p.parent.name for t in targets)\
        else ''
    speed_df.to_csv(p / (str(output_pref) + 'average_speed.csv'))
    order_df.to_csv(p / (str(output_pref) + 'order_parameter.csv'))

    # Data Processing: 6. Sorting the results by different conditions, e.g. drug treatment, time points, etc.
