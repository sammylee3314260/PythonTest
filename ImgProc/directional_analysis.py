import numpy as np
import os
from scipy.ndimage import gaussian_filter, sobel
import matplotlib.pyplot as plt
import tifffile

def compute_orientation_tensor(image, sigma = 0.2):
    # gradeint
    Ix = sobel(image, axis=1, mode = "reflect")
    Iy = sobel(image, axis=0, mode = "reflect")
    # tensor
    Ixx = gaussian_filter(Ix * Ix, sigma)
    Ixy = gaussian_filter(Ix * Iy, sigma)
    Iyy = gaussian_filter(Iy * Iy, sigma)

    # get direction map
    theta = 0.5 * np.arctan2(2 * Ixy, Ixx - Iyy)

    # get coherency
    lambda1 = 0.5 * (Ixx - Iyy + np.sqrt((Ixx - Iyy)**2 + 4*Ixy**2))
    lambda2 = 0.5 * (Ixx - Iyy - np.sqrt((Ixx - Iyy)**2 + 4*Ixy**2))
    coherency = (lambda1 - lambda2) / (lambda1 + lambda2 + 1e-10)

    return theta, coherency

if __name__ == "__main__":
    path = ""
    filename = ""
    if path=="":
        from skimage import data
        image = data.camera().astype(float)
    else:
        if not os.path.exists(path): print(f"path {path} not exist");exit()
        with tifffile.Tifffile(path+filename) as tifimg:
            image = tifimg.asarray()
    theta, coherency = compute_orientation_tensor(image)

    if 1: #plot
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.title("Orientation (radian)")
        plt.imshow(theta, cmap='hsv')
        plt.colorbar()
        
        plt.subplot(1, 2, 2)
        plt.title("Coherency")
        plt.imshow(coherency, cmap='gray')
        plt.colorbar()
        plt.tight_layout()
        plt.show()
