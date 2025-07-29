import numpy as np
import os
from scipy.ndimage import gaussian_filter,binary_closing
from skimage import img_as_float
from skimage.filters import scharr, frangi
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.segmentation import find_boundaries
import matplotlib.pyplot as plt
import tifffile
# import tensorflow as tf

# Try to add path for relative import
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    __package__ = 'ImgProc'
from . import image_display
from . import alpha_shape

def compute_orientation_tensor_2d(image, sigma:float = 0.2, eps:float = 1e-7,isMask = False):
    # gradeint
    # Ix = sobel(image, axis=1, mode = "reflect"); Iy = sobel(image, axis=0, mode = "reflect")
    Ix = scharr(image,axis=1); Iy = scharr(image,axis=0)
    # tensor
    # Structural_tensor J =
    #[[Ixx, Ixy];
    # [Ixy, Iyy]]
    Ixx = gaussian_filter(Ix * Ix, sigma)
    Ixy = gaussian_filter(Ix * Iy, sigma)
    Iyy = gaussian_filter(Iy * Iy, sigma)
    if isMask: return None, None, Ixx + Iyy # Just return Energy mask, reduce redundent calculation

    # get direction map
    # Minor eigenvector
    theta = 0.5 * np.arctan2(2 * Ixy, Ixx - Iyy)

    # get coherency
    # lambda1 = 0.5 * (Ixx + Iyy + np.sqrt((Ixx + Iyy)**2 + 4*(Ixx * Iyy - Ixy**2)))
    # lambda2 = 0.5 * (Ixx + Iyy - np.sqrt((Ixx + Iyy)**2 + 4*(Ixx * Iyy - Ixy**2)))
    # orientation_energy = lambda1 + lambda2
    # coherency = (lambda1 - lambda2) / (lambda1 + lambda2)
    delta = np.sqrt((Ixx + Iyy)**2 + 4*(Ixx * Iyy - Ixy**2))
    orientation_energy = Ixx + Iyy
    coherency = np.where(orientation_energy > eps,
                         (delta) / (orientation_energy + 1e-20),
                         0.0)
    return theta, coherency, orientation_energy

def hessian(image):
    image_float = img_as_float(image)
    return frangi(image_float,scale_range=(1,3),scale_step=0.5)

def cytoplasm_mask_fiber_2d(image, sigma:float = 0.2, eps:float = 1e-7):
    _, _, energy = compute_orientation_tensor_2d(image,sigma=sigma,isMask=True)
    return energy > eps

def compute_orientation_tensor_stack(image, sigma:float = 0.2, eps:float = 1e-7,isMask = False):
    shape = image.shape
    theta = []; coherence = [];energy = []
    for i in range(shape[0]):
        t, c, e = compute_orientation_tensor_2d(image[i],sigma=sigma,eps=eps,isMask=isMask)
        theta.append(t)
        coherence.append(c)
        energy.append(e)
    theta = np.stack(tuple(theta))
    coherence = np.stack(tuple(coherence))
    energy = np.stack(tuple(energy))
    return theta,coherence,energy

def cytoplasm_mask_fiber_stack(image, sigma:float = 0.2, eps:float = 1e-7,isMask = True):
    return compute_orientation_tensor_stack(image, sigma=sigma, isMask=isMask) > eps

def call_cyto_mask_fiber(path:str = "", filename:str = "",
                         sigma:float = 0.2,eps:float = 1e-7, dilation_width=3, alpha_thresh:float = 100,
                         display=True, manual_tune=True,):
    print("call_cyto_mask_fiber",path,filename)
    if path == "": print('empty path'); exit()
    if filename == "": print('empty filename'); exit()
    if not os.path.exists(os.path.join(path,filename)): print(f'path {os.path.join(path+filename)} not exist'); exit()
    with tifffile.TiffFile(os.path.join(path,filename)) as tiffimg:
        img = tiffimg.asarray()
    ndim = len(img.shape)
    if ndim == 2:
        _,_,energy = compute_orientation_tensor_2d(img,sigma=sigma,isMask=True)
    elif ndim == 3:
        _,_,energy = compute_orientation_tensor_stack(img,sigma=sigma,isMask=True)
    else: print(img.shape, "image shape is not 2D or 3D??");return
    while True:
        if eps == -1: break
        mask = energy > eps
        if ndim == 2:
            if display: image_display.mask_display(mask)
        elif ndim == 3:
            # binary closing
            padded = np.pad(mask,dilation_width,mode='constant',constant_values=False if mask.dtype=='bool' else 0)
            structure = np.ones((2*dilation_width+1,2*dilation_width+1,2*dilation_width+1),dtype=bool)
            closed = binary_closing(padded,structure=structure)
            final_mask = remove_small_objects(remove_small_holes(closed[dilation_width:-dilation_width,dilation_width:-dilation_width,dilation_width:-dilation_width],10000),10000)
            coords = np.column_stack(np.where(find_boundaries(final_mask)))
            print(f"Boundary points = {coords.shape}")
            if display: image_display.mask_display_3D(img,mask,final_mask)
        if not manual_tune:break
        input_eps = input(f'eps = {eps}')
        try: eps = float(input_eps)
        except: print(f'input eps = {input_eps} cannot convert to float'); break
    if ndim == 3: return final_mask
    return remove_small_holes(remove_small_objects(mask,1000),10000)

def batch_process_cyto_mask_fiber(path:str = "", filename:str|None = "", filter:str = "",sigma:int|None = 0.2,eps:float|None = None, dilation_width:int|None = 3):
    print(path)
    print(filename)
    # defalut value if no parameter is passed
    if sigma is None: sigma = 0.2
    if eps is None: eps = 1e-7
    if dilation_width is None: dilation_width = 3
    if path == "": print('empty path'); exit()
    if not os.path.exists(path): print(f'path {path+filename} not exist'); exit()

    if filename == "" or filename is None:
        filelist = os.listdir(path)
        print(filelist)
        filelist = [f for f in filelist if f.find('.tif')!=-1] # filter tif file only
        if filter != "":filelist = [f for f in filelist if f.find(filter)!=-1] # filter tif file only
        print(filelist)
        for f in filelist:
            print(f'filename = {f}')
            np.save(os.path.join(path,f.split('.tif')[0]+'_cytomask.npy'),
                    call_cyto_mask_fiber(path, f, sigma=sigma, eps=eps, display=True, dilation_width=dilation_width))
    else:
        print(f'filename == {filename}')
        print(os.path.join(path,filename.split('.tif')[0]+'_cytomask.npy'))
        np.save(os.path.join(path,filename.split('.tif')[0]+'_cytomask.npy'),
                call_cyto_mask_fiber(path, filename, sigma=sigma, eps=eps, display=True, dilation_width=dilation_width))



'''
def compute_orientation_tensor_tf_3d(image, sigma:float = 0.2, eps:float = 1e-7,isMask = False):
    # If axes are 'ZYX'
    dz = (image[2:,:,:]-image[:-2,:,:]) / 2; dz = tf.pad(dz,[[1,1],[0,0],[0,0]])
    dy = (image[:,2:,:]-image[:,:-2,:]) / 2; dy = tf.pad(dy,[[0,0],[1,1],[0,0]])
    dx = (image[:,:,2:]-image[:,:,:-2]) / 2; dx = tf.pad(dx,[[0,0],[0,0],[1,1]])
    
    Jxx = dx * dx;Jxx = tf.convert_to_tensor(gaussian_filter(Jxx.numpy(), sigma=sigma))
    Jxy = dx * dy;Jxy = tf.convert_to_tensor(gaussian_filter(Jxy.numpy(), sigma=sigma))
    Jxz = dx * dz;Jxz = tf.convert_to_tensor(gaussian_filter(Jxz.numpy(), sigma=sigma))
    Jyy = dy * dy;Jyy = tf.convert_to_tensor(gaussian_filter(Jyy.numpy(), sigma=sigma))
    Jyz = dy * dz;Jyz = tf.convert_to_tensor(gaussian_filter(Jyz.numpy(), sigma=sigma))
    Jzz = dz * dz;Jzz = tf.convert_to_tensor(gaussian_filter(Jzz.numpy(), sigma=sigma))

    J = tf.stack([
        tf.stack([Jxx,Jxy,Jxz],axis=-1),
        tf.stack([Jxy,Jyy,Jyz],axis=-1),
        tf.stack([Jxz,Jyz,Jzz],axis=-1),
    ],axis=-2)
    
    eigvals, eigvecs = tf.linalg.eigh(J[0])
    principal_vector = eigvecs[...,-1]
    energy = eigvals[...,-1]+eigvals[...,-2]+eigvals[...,-3]
    coherence = (eigvals[...,-1]-eigvals[...,-2])/(energy + 1e-10)
    viewer = napari.Viewer()
    viewer.add_image(coherence)
    napari.run()
    return principal_vector,energy,coherence
'''

if __name__ == "__main__":
    # path = "/mnt/SammyRis/Sammy/20250617_tiff/max_nucleus/"; filename = "2025-06-17-10AWT_Ctrl_001_max_C3.tif"
    path = "/mnt/SammyRis/Sammy/2025072021_exp_recov_max_proj/"
    filename="2025-07-21_10AWT_Ctrl_01_63x_002_max_C1.tif"

    if not os.path.exists(path): print(f"path {path} not exist");exit()
    with tifffile.TiffFile(os.path.join(path+filename)) as tifimg:
        image = tifimg.asarray()
    energy_thresh = 2e-4
    
    # hessian_img = hessian(image)

    if 1: #plot
        theta, coherency, energy = compute_orientation_tensor_2d(image,sigma=2,eps=energy_thresh)
        image_display.analy_display(image,theta,coherency,energy)
        # plt.show()

    if 0:
        mask = cytoplasm_mask_fiber_2d(image,sigma=1,eps = energy_thresh)
        image_display.mask_display(mask)

    if 0: # plot 3D stack analy
        call_cyto_mask_fiber(path, filename,sigma=1, eps=1e-6,dilation_width=2,manual_tune=True)