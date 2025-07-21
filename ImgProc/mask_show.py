import os
import numpy as np
import napari
import tifffile
from scipy.ndimage import find_objects, binary_dilation
from skimage.segmentation import find_boundaries

def label_boundary_fast(label_mask, thicken:bool = True):
    boundary_mask = np.zeros_like(label_mask)
    slices = find_objects(label_mask)
    for lbl, slc in enumerate(slices,start=1):
        if slc is None: continue
        region = label_mask[slc] == lbl
        boundary = find_boundaries(region,mode='outer')
        if thicken: boundary = binary_dilation(boundary, iterations=1)
        boundary_mask[slc] = np.where(boundary, lbl,boundary_mask[slc])
    return boundary_mask

if __name__ == "__main__":
    path = '/mnt/SammyRis/Sammy/20250617_tiff/'
    maskfile = '2025-06-17-10AWT_Hypo33_003_max_C1_cytomask.npy'
    file = '2025-06-17-10AWT_Hypo33_003_max_C1.tif'
    with tifffile.TiffFile(os.path.join(path,file)) as tifimg:
        img = tifimg.asarray()
    viewer = napari.Viewer()
    viewer.add_image(img, name = 'Raw Image', colormap = 'gray', contrast_limits = (np.percentile(img,1), np.percentile(img,99)))

    mask = np.load(os.path.join(path,maskfile))
    if mask.dtype == 'bool':
        mask = np.where(mask,1,0)

    viewer.add_labels(label_boundary_fast(mask),name = 'mask boundary')
    napari.run()