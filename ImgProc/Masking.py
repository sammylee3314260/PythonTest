import numpy as np
import matplotlib.pyplot as plt
from skimage import io, measure
from skimage.filters import gaussian, threshold_otsu
import tifffile
import os
from skimage.morphology import remove_small_holes, remove_small_objects, closing, disk
import cv2

path = 'C:\\Users\\pathaklab\\Box\\HZ_Sammy_PathakLab\\Zeiss 880 LSM Images\\2025-03-28\\tiff\\'
file = '2025-03-28-10AWT_Bleb_001_max_C1.tif'

if not os.path.exists(path+file):
    print('file not exist')
    exit()

with tifffile.TiffFile(path+file) as tifimg:
    # original image array read-only
    img = tifimg.asarray()
    img.flags.writeable = False

    # otsu threshold
    otsu_thresh = threshold_otsu(img)
    binary_mask = img > otsu_thresh

    #fill hole/ fill small
    decrease_ratio = 0.3
    while True:
        decreased_otsu = otsu_thresh*decrease_ratio
        decreased_binary_mask = img > decreased_otsu
        cleaned = remove_small_objects(decreased_binary_mask, min_size=50000)
        cleaned = remove_small_holes(cleaned, area_threshold= 200000)

        label_mask = measure.label(cleaned)
        contours = measure.find_contours(cleaned)
        fig, ax = plt.subplots()
        ax.imshow(img,cmap = 'gray')
        print('how many contours:', len(contours))
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], linewidth = 1, color = 'cyan')
        ax.axis('off')
        plt.tight_layout()
        plt.show()
        decrease_ratio = 0
        manual_change_ratio = False # whether you want to manually change threshold
        if manual_change_ratio:
            ans = input('decrease ratio = '+str(decrease_ratio)+' enter new:')
            decrease_ratio = float(ans)
        if decrease_ratio == -1:
            print(file+'no file saved')
            break
        if decrease_ratio == 0:
            #save mask to numpy file
            #np.save(path+file.split('.tif')[0]+'_plasm_mask.npy', cleaned)

            #save img with contour to jpg file
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            #contours_cv = [np.array(contour[:,[1,0]],dtype = np.int32) for contour in contours]
            print(type(cleaned))
            contours_cv,hierarchy = cv2.findContours(cleaned.copy().astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            print(contours_cv)
            cv2.drawContours(rgb,contours_cv,-1,(0,255,0),2)
            #plt.imshow(rgb)
            #plt.show()
            #cv2.imwrite(path+file.split('.tif')[0]+'_plasm_mask.jpg',rgb)
            break

    if decrease_ratio == -1: exit()
