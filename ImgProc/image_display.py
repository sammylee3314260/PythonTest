import os
import numpy as np
import napari
import tifffile
from scipy.ndimage import find_objects, binary_dilation, binary_closing
from skimage.segmentation import find_boundaries
import matplotlib.pyplot as plt
from skimage.morphology import remove_small_objects, remove_small_holes
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    __package__ = 'ImgProc'
from . import directional_analysis

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

def analy_display(image,theta,coherency,energy):
    plt.figure(figsize=(10,15))
    plt.subplot(2,3,1)
    plt.title("Orientation (radian)")
    plt.imshow(theta, cmap='hsv')
    plt.axis('off');plt.tight_layout();plt.colorbar()
        
    plt.subplot(2,3, 2)
    plt.title("Coherency")
    plt.imshow(coherency, cmap='gray',vmin=0,vmax=np.percentile(coherency,95))
    plt.colorbar();plt.axis('off');plt.tight_layout()

    plt.subplot(2,3, 3)
    plt.title("Vector")
    Y, X = np.meshgrid(np.arange(theta.shape[0]),np.arange(theta.shape[1]),indexing='ij')
    U = np.cos(theta); V = np.sin(theta)
    plt.imshow(image,cmap='gray',vmin=np.percentile(image,1),vmax=np.percentile(image,99))
    plt.quiver(X[::10,::10],Y[::10,::10],U[::10,::10],V[::10,::10],color='red',scale=50)
    plt.colorbar(); plt.axis('off'); plt.tight_layout()
        
    plt.subplot(2,3, 4)
    plt.title("Energy")
    plt.imshow(energy,cmap='gray',vmin=np.percentile(energy,0),vmax=np.percentile(energy,99))
    plt.colorbar();plt.axis('off');plt.tight_layout()

    plt.subplot(2,3,5)
    plt.title("Original")
    plt.imshow(image, cmap='gray')
    plt.axis('off');plt.colorbar()

    # plt.subplot(2,3, 6)
    #plt.title("Frangi")
    #plt.imshow(hessian_img, cmap='gray',vmin=np.percentile(hessian_img,1),vmax=np.percentile(hessian_img,99))
    #plt.colorbar();plt.axis('off');plt.tight_layout()

    plt.show()

def mask_display(mask):
    plt.figure()

    plt.subplot(1,3,1);plt.axis('off');plt.tight_layout()
    plt.imshow(mask)
    plt.subplot(1,3,2);plt.axis('off');plt.tight_layout()
    plt.imshow(remove_small_holes(mask))
    plt.subplot(1,3,3);plt.axis('off');plt.tight_layout()
    plt.imshow(remove_small_holes(remove_small_objects(mask),10000))

    plt.show()

def mask_display_3D(image, mask,final_mask,alpha_thresh:float = 100):
    napari_plot = 1
    if napari_plot: viewer = napari.Viewer()
    if napari_plot: viewer.add_image(image,name='Raw Image',colormap='gray',contrast_limits=(np.percentile(image,1),np.percentile(image,99)))
    if napari_plot: viewer.add_image(mask,name='mask',colormap='gray',contrast_limits=(0,1),opacity=0.3)
    # if napari_plot: viewer.add_image(remove_small_holes(mask,10000),name='mask remove holes',colormap='gray',contrast_limits=(0,1),opacity=0.3)
    # if napari_plot: viewer.add_image(remove_small_objects(remove_small_holes(mask,10000)),name='mask remove hole -> obj',colormap='gray',contrast_limits=(0,1),opacity=0.3)
    # final_mask = remove_small_objects(remove_small_holes(closed))
    if napari_plot: viewer.add_image(final_mask,name='Binary closing',colormap='green',opacity=0.5,contrast_limits=(0,1))
    # viewer.add_image(alpha_shape.alpha_shape(coords,alpha_thresh),name='Alpha shape',colormap='Red',opacity=0.5,contrast_limits=(0,1))
    if napari_plot: napari.run()

if __name__ == "__main__":
    path = '/mnt/SammyRis/Sammy/YAP_Actin_lamAC_3D_masks_40x/'
    maskfile = '10A_5kPa_Ctrl_POS48h_40xoil_C1_cytomask.npy'
    file = '10A_5kPa_Ctrl_POS48h_40xoil_C1.tif'
    with tifffile.TiffFile(os.path.join(path,file)) as tifimg:
        img = tifimg.asarray()
    viewer = napari.Viewer()
    viewer.add_image(img, name = 'Raw Image', colormap = 'gray', contrast_limits = (np.percentile(img,1), np.percentile(img,99)))

    # mask = np.load(os.path.join(path,maskfile))
    _,_,energy = directional_analysis.compute_orientation_tensor_stack(img,1,1e-7)
    mask = energy > 1e-5
    if mask.dtype == 'bool':
        mask = np.where(mask,1,0)

    viewer.add_labels(label_boundary_fast(mask),name = 'mask boundary')
    napari.run()

    # path = /mnt/SammyRis/Sammy/2025072021_max_proj/
    # 2025-07-20_10AWT_Ctrl_02_63x_001_max_C1.tif 2025-07-20_10AWT_Ctrl_02_63x_001_max_C1_cytomask.npy