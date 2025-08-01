import tifffile
import cv2

import numpy as np

import matplotlib.pyplot as plt
import os
import math

path = '/mnt/SammyRis/Sammy/2025072021_exp_recov_max_proj/'

if not os.path.exists(os.path.join(path+'jpeg')):
    os.mkdir(path+'jpeg')

filelist = os.listdir(path)
#filelist = ['MAX_2025-03-28-10AWT_Ctrl_001_405_nuc_488_pax_555_actin_647_NMIIa_C2.tif']
filenames = []
img = 0
temp = 0
bin = range(256)
freq = np.zeros((256,),dtype = np.int64) #[range = 0~1<<32-1]
filter = 'max_C2.tif'
print(filter)
for file in filelist:
    if not file.endswith('.tif') and not file.endswith('.tiff') or file.find(filter)==-1:
        # print(file)
        # filer 'max' to only select max proj
        continue
    filenames.append(file)
    # print(filenames)
    #with tifffile.TiffFile(path+file) as tifimg:
    with tifffile.TiffFile(os.path.join(path, file)) as tifimg:
        #print(file)
        temp = tifimg.asarray()
        if len(filenames) == 1:
            img = np.expand_dims(temp, axis=2)
            #print('page :', tifimg.pages[0].tags[283])
            #print('value:', tifimg.pages[0]._gettags()[11][1].value)
            #print('dtype:',tifimg.pages[0]._gettags()[6][1].dtype_name)
            #exit()
        else:
            img = np.dstack((img, temp))
        #print(freq.shape, np.histogram(temp,bins=256,range=(0,256))[0].shape)

        #calculation of histogram frequency
        freq += np.histogram(temp, bins=256,range=(0,256))[0]
filenum = len(filenames)

#print(img.max(axis=0).max(axis=0))
# plot histogram freq
print(filenames)
plt.bar(bin, freq)
print('finish plt')
plt.title("Histogram of original image")
plt.xlabel("Pixel value")
plt.ylabel("Frequency")
plt.show(block=False)

# show all images
column = math.ceil(math.sqrt(len(filenames)))
row = math.ceil(len(filenames)/column)
f, ax = plt.subplots(row, column)
edge = min(f.get_size_inches())
# f.set_size_inches(edge*column, edge*row)
#f.set_dpi(f.get_dpi()*max(row,column))
print('column = ', column, 'row = ', row)

#change the gray scale for all
max_intensity = int(np.percentile(img,99.75))

while True:
    # change the gray scale for all
    print('max_intensity = ', max_intensity)
    stretched = np.clip(img, 0, max_intensity)
    stretched = stretched / (max_intensity + 1e-5) * 255
    stretched = np.clip(stretched, 0, 255).astype(np.uint8)

    if column * row == 1:
        ax.imshow(img[:, :, 0],cmap = 'gray')
        ax.axis('off')
        plt.tight_layout()
    else:
        for i in range(row*column):
            #print('i // column', i // column, 'i % column', i % column)
            #print(type(ax))
            #ax.imshow(img[:, :, i])
            if i < filenum:
                ax[i//column][i%column].imshow(stretched[:,:,i],cmap = 'gray')
            ax[i//column][i%column].axis('off')
            plt.tight_layout()
    f.show()


    # manual change threshold
    manual_change_max = True
    if manual_change_max:
        ans = input('max intensity = ' + str(max_intensity) + ' enter new:')
        try: max_intensity = int(ans)
        except: print("cannot convert to int"); break
    else:
        break
    if max_intensity == 0:
        break
    if max_intensity == -1:
        print("no image saved")
        exit()

#save images
for i in range(filenum):
    rgb = cv2.cvtColor(stretched[:,:,i], cv2.COLOR_GRAY2BGR)
    cv2.imwrite(os.path.join(path,'jpeg',filenames[i].split(".tif")[0]+".jpg"), rgb)

f.savefig(os.path.join(path,'jpeg',filter+'_box.jpeg'),transparent=True)