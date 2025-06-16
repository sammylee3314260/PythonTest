import numpy as np
import matplotlib.pyplot as plt
import tifffile
import czifile
import os

path = 'C:\\Users\\pathaklab\\Box\\HZ_Sammy_PathakLab\\Zeiss 880 LSM Images\\2025-03-28\\'
if not os.path.exists(path):
    print('path not exist')
    exit()
if not os.path.exists(path+'tiff\\'):
    os.mkdir(path+'tiff')
files = os.listdir(path)
filepath = './*.czi'
for file in files:
    if not file.endswith('.czi'):
        continue
    cziimg = czifile.CziFile(path+file)
    print(cziimg.axes,cziimg.shape) # (1, 1, 4, 1, 17, 1024, 1024, 1)
    channels = cziimg.shape[2]
    filename = file.split('405')[0]
    for channel in range(channels):
        image = cziimg.asarray()[0,0,channel,0,:,:,:,0]
        #check scale factor
        scalefactorstrings = cziimg.metadata().split('<Distance Id="X">')[1].split('</Distance>')
        scalefactorX = float(scalefactorstrings[0].split('</Value>')[0].split('<Value>')[1])
        scalefactorY = float(scalefactorstrings[1].split('</Value>')[0].split('<Value>')[1])
        scalefactorZ = float(scalefactorstrings[2].split('</Value>')[0].split('<Value>')[1])
        tifffile.imwrite(path+'tiff\\'+filename+'C'+str(channel)+'.tif',image,shape=image.shape,
                         dtype=image.dtype, resolution=(1/scalefactorZ, 1/scalefactorX, 1/scalefactorY),
                         metadata={'unit':'um','axes':'ZXY'})
        #max projection
        maxproj = np.empty((image.shape[1],image.shape[2]),dtype=image.dtype)
        for i in range(image.shape[1]): # x index
            for j in range(image.shape[2]):  # y index
                maxproj[i,j] = image[:,i,j].max()
        tifffile.imwrite(path + 'tiff\\' + filename + 'max_C' + str(channel) + '.tif', image, shape=image.shape,
                         dtype=image.dtype, resolution=(1/scalefactorX, 1/scalefactorY),
                         metadata={'unit': 'um', 'axes': 'XY'})

"""
To get scaling factors
>>> file.metadata()
'<Scaling>
      <Items>
        <Distance Id="X">
          <Value>1.3178822554981574e-007</Value>
        </Distance>
        <Distance Id="Y">
          <Value>1.3178822554981574e-007</Value>
        </Distance>
        <Distance Id="Z">
          <Value>2.4999999999999999e-007</Value>
        </Distance>
      </Items>
    </Scaling>
'
"""