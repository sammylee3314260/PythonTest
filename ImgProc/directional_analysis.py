import numpy as np
import os
from scipy.ndimage import gaussian_filter, sobel
from skimage.filters import scharr, frangi
from skimage import img_as_float
import matplotlib.pyplot as plt
import tifffile

def compute_orientation_tensor(image, sigma = 0.2):
    # gradeint
    # Ix = sobel(image, axis=1, mode = "reflect"); Iy = sobel(image, axis=0, mode = "reflect")
    Ix = scharr(image,axis=1); Iy = scharr(image,axis=0)
    # tensor
    Ixx = gaussian_filter(Ix * Ix, sigma)
    Ixy = gaussian_filter(Ix * Iy, sigma)
    Iyy = gaussian_filter(Iy * Iy, sigma)

    # get direction map
    theta = 0.5 * np.arctan2(2 * Ixy, Ixx - Iyy)

    # get coherency
    lambda1 = 0.5 * (Ixx - Iyy + np.sqrt((Ixx - Iyy)**2 + 4*Ixy**2))
    lambda2 = 0.5 * (Ixx - Iyy - np.sqrt((Ixx - Iyy)**2 + 4*Ixy**2))
    orientation_energy = lambda1 + lambda2
    eps = 0
    coherency = np.where(orientation_energy > eps,
                         (lambda1 - lambda2) / (lambda1 + lambda2 + 1e-10),
                         0.0)
    return theta, coherency, orientation_energy

def hessian(image):
    image_float = img_as_float(image)
    return frangi(image_float,scale_range=(1,3),scale_step=0.5)

if __name__ == "__main__":
    # path = "/mnt/SammyRis/Active/lab/Sammy/20250617_tiff/"; filename = "2025-06-17-10AWT_Ctrl_001_max_C2.tif"
    path = "/mnt/SammyRis/Active/lab/Sammy/YAP_Actin_lamAC_3D_masks_40x/"; filename = "10A_5kPa_Ctrl_POS24h_40xoil_C1_avg15_17.tif"
    if path=="":
        from skimage import data
        image = data.camera().astype(float)
    else:
        if not os.path.exists(path): print(f"path {path} not exist");exit()
        with tifffile.TiffFile(path+filename) as tifimg:
            image = tifimg.asarray()
    theta, coherency, energy = compute_orientation_tensor(image,1)

    hessian_img = hessian(image)

    if 1: #plot
        plt.figure(figsize=(10,10))
        plt.subplot(2,2,1)
        plt.title("Orientation (radian)")
        plt.imshow(theta, cmap='hsv')
        plt.axis('off')
        plt.tight_layout()
        plt.colorbar()
        
        plt.subplot(2, 2, 2)
        plt.title("Coherency")
        plt.imshow(coherency, cmap='gray',vmin=np.percentile(coherency,1),vmax=np.percentile(coherency,75))
        plt.colorbar()
        plt.axis('off')
        plt.tight_layout()

        plt.subplot(2, 2, 3)
        plt.title("Vector")
        Y, X = np.meshgrid(np.arange(theta.shape[0]),np.arange(theta.shape[1]),indexing='ij')
        U = np.cos(theta); V = np.sin(theta)
        plt.imshow(image,cmap='gray',vmin=np.percentile(image,1),vmax=np.percentile(image,99))
        plt.quiver(X[::10,::10],Y[::10,::10],U[::10,::10],V[::10,::10],color='red')
        plt.colorbar()
        plt.tight_layout()
        
        plt.subplot(2, 2, 4)
        plt.title("Energy")
        plt.imshow(energy,cmap='gray',vmin=np.percentile(energy,0),vmax=np.percentile(energy,100))
        plt.colorbar()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.title("Original")
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.title("Frangi")
        plt.imshow(hessian_img, cmap='gray',vmin=np.percentile(hessian_img,1),vmax=np.percentile(hessian_img,99))
        plt.colorbar()
        plt.axis('off')
        plt.tight_layout()
        plt.show()
