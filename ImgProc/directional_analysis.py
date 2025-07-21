import numpy as np
import os
from scipy.ndimage import gaussian_filter, sobel
from skimage import img_as_float
from skimage.filters import scharr, frangi
from skimage.morphology import remove_small_objects, remove_small_holes
import matplotlib.pyplot as plt
import tifffile

def compute_orientation_tensor(image, sigma:float = 0.2, eps:float = 1e-7,isMask = False):
    # gradeint
    # Ix = sobel(image, axis=1, mode = "reflect"); Iy = sobel(image, axis=0, mode = "reflect")
    Ix = scharr(image,axis=1); Iy = scharr(image,axis=0)
    # tensor
    Ixx = gaussian_filter(Ix * Ix, sigma)
    Ixy = gaussian_filter(Ix * Iy, sigma)
    Iyy = gaussian_filter(Iy * Iy, sigma)
    if isMask: return None, None, Ixx + Iyy # Just return Energy mask, reduce redundent calculation

    # get direction map
    theta = 0.5 * np.arctan2(2 * Ixy, Ixx - Iyy)

    # get coherency
    # lambda1 = 0.5 * (Ixx + Iyy + np.sqrt((Ixx + Iyy)**2 + 4*(Ixx * Iyy - Ixy**2)))
    # lambda2 = 0.5 * (Ixx + Iyy - np.sqrt((Ixx + Iyy)**2 + 4*(Ixx * Iyy - Ixy**2)))
    # orientation_energy = lambda1 + lambda2
    delta = np.sqrt((Ixx + Iyy)**2 + 4*(Ixx * Iyy - Ixy**2))
    orientation_energy = Ixx + Iyy
    coherency = np.where(orientation_energy > eps,
                         (delta) / (orientation_energy + 1e-20),
                         0.0)
    return theta, coherency, orientation_energy

def hessian(image):
    image_float = img_as_float(image)
    return frangi(image_float,scale_range=(1,3),scale_step=0.5)

def cytoplasm_mask_fiber(image, sigma:float = 0.2, eps:float = 1e-7):
    _, _, energy = compute_orientation_tensor(image,isMask=True)
    return energy > eps

def call_cyto_mask_fiber(path:str = "", filename:str = "",eps = 1e-7, display=False):
    if path == "": print('empty path'); exit()
    if filename == "": print('empty filename'); exit()
    if not os.path.exists(os.path.join(path,filename)): print(f'path {os.path.join(path+filename)} not exist'); exit()
    with tifffile.TiffFile(os.path.join(path,filename)) as tiffimg:
        img = tiffimg.asarray()
    while True:
        if eps == -1: break
        mask = cytoplasm_mask_fiber(img,eps=eps)
        if display:
            plt.figure()
            plt.subplot(1,3,1);plt.axis('off');plt.tight_layout()
            plt.imshow(mask)
            plt.subplot(1,3,2);plt.axis('off');plt.tight_layout()
            plt.imshow(remove_small_holes(mask))
            plt.subplot(1,3,3);plt.axis('off');plt.tight_layout()
            plt.imshow(remove_small_holes(remove_small_objects(mask,1000),10000))
            plt.show(block=False)
        input_eps = input(f'eps = {eps}')
        try: eps = float(input_eps)
        except: print(f'input eps = {input_eps} cannot convert to float'); break
    return remove_small_holes(remove_small_objects(mask,1000),10000)

def batch_process_cyto_mask_fiber(path:str = "", filename:str = "", filter:str = "",eps = 1e-5):
    print(path)
    print(filename)
    if path == "": print('empty path'); exit()
    if not os.path.exists(path): print(f'path {path+filename} not exist'); exit()

    if filename == "":
        filelist = os.listdir(path)
        print(filelist)
        filelist = [f for f in filelist if f.find('.tif')!=-1] # filter tif file only
        if filter != "":filelist = [f for f in filelist if f.find(filter)!=-1] # filter tif file only
        print(filelist)
        for f in filelist:
            print(f'filename = {f}')
            np.save(os.path.join(path,f.split('.tif')[0]+'_cytomask.npy'),
                    call_cyto_mask_fiber(path,f,eps= eps,display=True))
    else:
        print(f'filename == {filename}')
        print(os.path.join(path,filename.split('.tif')[0]+'_cytomask.npy'))
        np.save(os.path.join(path,filename.split('.tif')[0]+'_cytomask.npy'),
                call_cyto_mask_fiber(path, filename, display=True))


if __name__ == "__main__":
    path = "/mnt/SammyRis/Active/lab/Sammy/20250617_tiff/"; filename = "2025-06-17-10AWT_Ctrl_001_max_C1.tif"
    # path = "/mnt/SammyRis/Active/lab/Sammy/YAP_Actin_lamAC_3D_masks_40x/"; filename = "10A_5kPa_Ctrl_POS24h_40xoil_C1_avg15_17.tif"
    if path=="":
        from skimage import data
        image = data.camera().astype(float)
    else:
        if not os.path.exists(path): print(f"path {path} not exist");exit()
        with tifffile.TiffFile(path+filename) as tifimg:
            image = tifimg.asarray()
    energy_thresh = 1e-4
    theta, coherency, energy = compute_orientation_tensor(image,0.2,energy_thresh)

    hessian_img = hessian(image)

    if 0: #plot

        plt.figure(figsize=(10,15))
        plt.subplot(2,3,1)
        plt.title("Orientation (radian)")
        plt.imshow(theta, cmap='hsv')
        plt.axis('off')
        plt.tight_layout()
        plt.colorbar()
        
        plt.subplot(2,3, 2)
        plt.title("Coherency")
        plt.imshow(coherency, cmap='gray',vmin=np.percentile(coherency,1),vmax=np.percentile(coherency,75))
        plt.colorbar()
        plt.axis('off')
        plt.tight_layout()

        plt.subplot(2,3, 3)
        plt.title("Vector")
        Y, X = np.meshgrid(np.arange(theta.shape[0]),np.arange(theta.shape[1]),indexing='ij')
        U = np.cos(theta); V = np.sin(theta)
        plt.imshow(image,cmap='gray',vmin=np.percentile(image,1),vmax=np.percentile(image,99))
        plt.quiver(X[::10,::10],Y[::10,::10],U[::10,::10],V[::10,::10],color='red')
        plt.colorbar()
        plt.axis('off')
        plt.tight_layout()
        
        plt.subplot(2,3, 4)
        plt.title("Energy")
        plt.imshow(energy,cmap='gray',vmin=np.percentile(energy,0),vmax=np.percentile(energy,99))
        plt.colorbar()
        plt.axis('off')
        plt.tight_layout()

        plt.subplot(2,3,5)
        plt.title("Original")
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.colorbar()

        plt.subplot(2,3, 6)
        plt.title("Frangi")
        plt.imshow(hessian_img, cmap='gray',vmin=np.percentile(hessian_img,1),vmax=np.percentile(hessian_img,99))
        plt.colorbar()
        plt.axis('off')
        plt.tight_layout()
        # plt.show()

    if 1:
        plt.figure()
        mask = energy > energy_thresh
        plt.subplot(1,3,1)
        plt.imshow(mask)
        plt.axis('off')
        plt.tight_layout()
        plt.subplot(1,3,2)
        plt.imshow(remove_small_holes(mask))
        plt.axis('off')
        plt.tight_layout()
        plt.subplot(1,3,3)
        plt.imshow(remove_small_holes(remove_small_objects(mask),10000))
        plt.axis('off')
        plt.tight_layout()
        plt.show()


