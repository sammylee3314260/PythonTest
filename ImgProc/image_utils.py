############################
# image processing tools
# - czi to tif
# - tif to jpeg
# - tif to npy
# - npy + mask to quant
############################

import numpy as np
import czifile
import tifffile
import os
from typing import Tuple,Dict,Generator, List
from itertools import product
import matplotlib.pyplot as plt
import aicspylibczi
import math
import cv2
import xml.etree.ElementTree as ET

def iter_multi_axes(axes:str = 'BVCTZYX0',# = "",
                    shape:Tuple[int,...] = (1, 1, 4, 1, 16, 1024, 1024, 1),# = None,
                    iter_axes:Dict[str,...] = {'S','T','C'},
                    pres_axes:Dict[str,...] = {'X','Y','Z'},
                    file_pref:str = "img_") ->(
                    Generator[tuple[tuple[int|None,...],int|None,str,str],None,None]):

    axes_map = {ax:i for i,ax in enumerate(axes)}
    # check axes for filename
    is_multi_scene, is_multi_time, is_multi_channel = False,False,False
    if 'S' in axes_map and shape[axes_map['S']] > 1: is_multi_scene = True
    if 'T' in axes_map and shape[axes_map['T']] > 1: is_multi_time = True
    if 'C' in axes_map and shape[axes_map['C']] > 1: is_multi_channel = True

    # check if the to-be-iterated axes exist,
    # if so add to iter_axes
    loop_axes = []
    loop_range = []
    for ax in iter_axes:
        if ax in axes_map:
            print(ax, shape[axes_map[ax]])
            loop_axes.append(ax)
            loop_range.append(range(shape[axes_map[ax]]))
    Z_ind = None
    if 'Z' in axes_map and 'Z' in pres_axes: # check if this img is 3D, if so return index of Z
        pres_ind = []
        cnt = 0
        for ax in axes_map:
            if ax == 'Z': Z_ind = cnt; break
            elif ax in pres_axes: cnt += 1

    for index_combo in product(*loop_range):
        slicer = [0] * len(shape) # index = dimension of the img
        for ax, idx in zip(loop_axes, index_combo): slicer[axes_map[ax]] = idx
        for ax in pres_axes:
            if ax in axes_map and not ax in loop_axes: slicer[axes_map[ax]] = slice(None)
            # This is a very important line, slice(None) = [:] so we can take the whole stack!
        out_filename = (file_pref[0:-1]
                        + ('_S' + str(slicer[axes_map.get('S')]) if is_multi_scene else '')
                        + ('_T' + str(slicer[axes_map.get('T')]) if is_multi_time else '')
                        + ('_C' + str(slicer[axes_map.get('C')]) if is_multi_channel else '')
                        )
        out_file_max = (file_pref + 'max'
                        + ('_S' + str(slicer[axes_map.get('S')]) if is_multi_scene else '')
                        + ('_T' + str(slicer[axes_map.get('T')]) if is_multi_time else '')
                        + ('_C' + str(slicer[axes_map.get('C')]) if is_multi_channel else '')
                        )
        yield tuple(slicer), Z_ind, out_filename, out_file_max

def tiffw(path, img_array, axes, shape, file_pref,scalefactor):
    if file_pref[-1]!='_': file_pref = file_pref+'_' # this might have to be fine tuned
    for ind, Z_ind, out_filename, out_file_max in iter_multi_axes(axes,shape,file_pref=file_pref):
        print(out_filename)
        img = img_array[ind]
        # check scale factor
        '''
        resolution manipulation:
            resolution code: X: 282, Y:283
            in czi file the resolution unit is meter/pixel
            in tiff file the resolution unit is pixel/resUnit (Code 296)
        '''
        tifffile.imwrite(os.path.join(path, 'tiff', out_filename + '.tif'), img,
                        shape=img.shape,
                        imagej=True,
                        dtype=img.dtype,
                        software='ImageJ',
                        resolution=((int(1 / scalefactor['Y']), int(1e6)), (int(1 / scalefactor['X']), int(1e6))),
                        metadata={
                            'spacing': scalefactor['Z'] * int(1e6),
                            'unit': 'micron',
                            'axes': 'ZYX'})
        # max projection
        if (Z_ind is not None) and (img.ndim >= 3):
            maxproj = np.max(img, axis = Z_ind)
            tifffile.imwrite(os.path.join(path, 'tiff', out_file_max + '.tif'), maxproj,
                             shape=maxproj.shape,
                             imagej=True,
                             dtype=maxproj.dtype,
                             software='ImageJ',
                             resolution=((int(1 / scalefactor['Y']), int(1e6)), (int(1 / scalefactor['X']), int(1e6))),
                             # unit px/m
                             metadata={
                                 'unit': 'micron',
                                 'axes': 'YX'})

def cziwr(path:str = '', filelist:List[str] = [], prefix_splitter:str|None = '.czi'):
    if prefix_splitter is None: prefix_splitter = '.czi'
    if not os.path.exists(os.path.join(path,'tiff/')):
        os.mkdir(os.path.join(path,'tiff/'))
    for file in filelist:
        if not os.path.exists(os.path.join(path, file)):
            print(f'File {file} does not exist')
            continue
        if not file.lower().endswith('.czi'):
            print(f'File {file} is not a czi file')
            continue
        # axes, shape, img_array, file_pref, scalefactorX, scalefactorY, scalefactorZ = '',(), None,'',0,0,0
        with czifile.CziFile(os.path.join(path,file)) as cziimg:
            print(file)
            print(cziimg.axes, cziimg.shape)  # (1, 1, 4, 1, 17, 1024, 1024, 1)
            # What I usually care: scene(S), time (T), channel (C), x, y, z
            # cziimg.axes = 'BVSCTZYX0'
            # cziimg.shape = (1, 1, 2, 1, 3, 34, 1000, 1000, 1)
            axes = cziimg.axes
            shape = cziimg.shape
            file_pref = file.split(prefix_splitter)[0]
            scalefactorstrings = cziimg.metadata().split('<Distance Id="X">')[1].split('</Distance>')
            scalefactor={}
            scalefactor['X'] = float(scalefactorstrings[0].split('</Value>')[0].split('<Value>')[1])
            scalefactor['Y'] = float(scalefactorstrings[1].split('</Value>')[0].split('<Value>')[1])
            scalefactor['Z'] = float(scalefactorstrings[2].split('</Value>')[0].split('<Value>')[1])
            img_array = cziimg.asarray()
        tiffw(path, img_array, axes, shape, file_pref,scalefactor)
        '''
        for ind, Z_ind, out_filename, out_file_max in iter_multi_axes(axes,shape,file_pref=file_pref):
            img = img_array[ind]
            # check scale factor
            '
            resolution manipulation:
                resolution code: X: 282, Y:283
                in czi file the resolution unit is meter/pixel
                in tiff file the resolution unit is pixel/resUnit (Code 296)
            '
            tifffile.imwrite(os.path.join(path, 'tiff', out_filename + '.tif'), img,
                            shape=img.shape,
                            imagej=True,
                            dtype=img.dtype,
                            software='ImageJ',
                            resolution=((int(1 / scalefactorY), 1000000), (int(1 / scalefactorX), 1000000)),
                            metadata={
                                'spacing': scalefactorZ * 1000000,
                                'unit': 'micron',
                                'axes': 'ZYX'})
            # max projection
            if (Z_ind is not None) and (img.ndim >= 3):
                maxproj = np.max(img, axis = Z_ind)
                tifffile.imwrite(os.path.join(path, 'tiff', out_file_max + '.tif'), maxproj,
                                 shape=maxproj.shape,
                                 imagej=True,
                                 dtype=maxproj.dtype,
                                 software='ImageJ',
                                 resolution=((int(1 / scalefactorY), 1000000), (int(1 / scalefactorX), 1000000)),
                                 # unit px/m
                                 metadata={
                                     'unit': 'micron',
                                     'axes': 'YX'})
            '''

def callcziwr(path="",prefix_splitter:str|None = '.czi'):
    if not os.path.exists(path): print(f'path {path} not exists');return
    if prefix_splitter is None: prefix_splitter = '.czi'
    filelist = [f for f in os.listdir(path) if f.endswith('.czi')]
    print(path, filelist)
    cziwr(path,filelist=filelist,prefix_splitter=prefix_splitter)
    return
    #

def aics_cziwr(path:str = '', filelist:List[str] = [], prefix_splitter:str|None = '.czi'):
    if not os.path.exists(path): print(f"path {path} not exists"); return
    if prefix_splitter is None: prefix_splitter = '.czi'; print('No prefix_splitter, default is \'.czi\'')
    if not os.path.exists(os.path.join(path,'tiff/')): os.mkdir(os.path.join(path,'tiff/'))
    for file in filelist:
        if not os.path.exists(os.path.join(path, file)):
            print(f'File {file} does not exist')
            continue
        if not file.lower().endswith('.czi'):
            print(f'File {file} is not a czi file')
            continue
        cziimg = aicspylibczi.CziFile(os.path.join(path,file))
        shape = cziimg.size
        if shape == (0,0): print(f"This file {os.path.join(path,file)} has shape (0,0).\nThe file might be a main file of a group of split files. Cannot read!\n"); continue

        print(file,shape)
        scaling = cziimg.meta.find(".//Scaling")
        scalefactor = {}
        if scaling is not None:
            for dist in scaling.findall("Items/Distance"):
                axis = dist.attrib.get("Id")
                val = float(dist.find("Value").text)
                print(dist.attrib)
                # unit = dist.find("DefaultUnitFormat").text
                # scalefactor[axis] = (val,unit)
                scalefactor[axis] = val
                # print(axis, val, unit) # axes: 'X' 'Y' 'Z'
                print(axis, val)
        img_array, _ = cziimg.read_image()
        axes = cziimg.dims
        file_pref = file.split(prefix_splitter)[0]
        tiffw(path, img_array, axes, shape, file_pref, scalefactor)

def call_aics_cziwr(path:str = '',prefix_splitter:str|None = '.czi'):
    if not os.path.exists(path): print(f"path {path} not exists"); return
    filelist = os.listdir(path)
    aics_cziwr(path=path, filelist=filelist, prefix_splitter=prefix_splitter)

def tiff2jpeg(path = '', filelist = []):
    if not os.path.exists(path): print(f'path {path} not exists'); return
    if len(filelist) == 0: print('No files to process'); return
    if not os.path.exists(os.path.join(path, 'jpeg')):
        os.mkdir(os.path.join(path, 'jpeg'))
    xres = None
    yres = None
    img_list = []
    for file in filelist:
        with tifffile.TiffFile(os.path.join(path, file)) as tifimg:
            img_list.append(tifimg.asarray())
            # I want: um/px -> xres[1]/xres[0]
            if xres is None: xres = tifimg.pages[0].tags['XResolution'].value # the unit: px/um
            if yres is None: yres = tifimg.pages[0].tags['YResolution'].value
    img = np.stack(img_list)
    # test / debug
    bins = img.max()-img.min()
    freq = np.histogram(img,bins=bins,range=(img.min(),img.max()))[0]
    print(freq)
    filenum = len(filelist)

    '''
    plt.bar(range(img.min(),img.max()),freq)
    plt.title('Histogram of original images')
    plt.xlabel('Pixel value')
    plt.xlim(img.min(),img.max())
    plt.ylabel('Frequency')
    plt.show(block=False)
    '''
    # display all images
    row = math.ceil(math.sqrt(filenum/2))
    column = math.ceil(filenum/row)

    # column = math.ceil(math.sqrt(filenum))
    # row = math.ceil(filenum/column)
    
    f, ax = plt.subplots(row, column)
    print('column = ', column, 'row = ', row)
    max_intensity = int(np.percentile(img,99.75))
    while True:
        print(f'max_intensity = {max_intensity}')
        stretched = np.clip(img,0,max_intensity)
        stretched = stretched / (max_intensity+1e-5)*255
        stretched = np.clip(stretched,0,255).astype(np.uint8)
        if column*row == 1:
            ax.imshow(stretched[0],cmap='gray'); ax.axis('off'); plt.tight_layout()
        else:
            for i in range(row*column):
                if i < filenum:
                    ax[i//column][i%column].imshow(stretched[i],cmap='gray')
                ax[i//column][i%column].axis('off'); plt.tight_layout()
        f.show()
        # if you dont want to keep tuning intensity (ex. during testing) turn on this break
        # break
        ans = input('max_intensity = '+ str(max_intensity)+' Enter new:')
        try: max_intensity = int(ans)
        except: print('Cannot convert to int');break
    for i in range(filenum):
        temp = stretched[i]
        scalebar_um = 50
        scalebar_px = int(scalebar_um*xres[0]/xres[1])
        x = temp.shape[1] - 20
        y = temp.shape[0] - 20
        img_bgr = cv2.cvtColor(temp,cv2.COLOR_GRAY2BGR)
        cv2.rectangle(img_bgr, (x-scalebar_px,y),(x,y-int(temp.shape[0]/100)),(255,255,255),-1)
        text = f'{scalebar_um} um'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        thickness = 5
        (text_w,text_h) ,baseline = cv2.getTextSize(text=text,fontFace=font,fontScale=font_scale,thickness=thickness)
        text_x = x-scalebar_px//2 - text_w//2
        text_y = y-int(temp.shape[0]/100)-10
        cv2.putText(img_bgr, text, (text_x,text_y),
                    fontFace=font, fontScale=font_scale, color=(255,255,255),thickness=thickness,lineType=cv2.LINE_AA)
        cv2.imwrite(os.path.join(path, 'jpeg', (filelist[i].split('.tif')[0]+'.jpeg')),img_bgr)
        # if you just want to test the effect of cv2 plotting, turn on this exit
        # exit()

    
def calltiff2jpeg(path,filter):
    if filter is None: filter = '.tif'
    if not os.path.exists(path): print(f'path {path} not exists'); return
    filelist = os.listdir(path)
    filelist = [f for f in filelist if ((f.endswith('.tif') or f.endswith('.tiff')) and f.find(filter)!=-1)]
    tiff2jpeg(path, filelist)


if __name__ == '__main__':
    path = '/mnt/box/HyperOsmo/250710_CD7_Nuc_Vol_Day1/'
    filnames = '10a_5kPa_Exposure-01(1).czi'
    # aics_cziwr(path, [filnames],prefix_splitter='.czi')
    # call_aics_cziwr(path)