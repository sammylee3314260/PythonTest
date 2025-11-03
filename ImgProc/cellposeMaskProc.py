import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tifffile
# from dask.array import outer
# from dask.array.reshape import contract_tuple
from scipy.ndimage import binary_fill_holes, find_objects, binary_dilation,distance_transform_edt, gaussian_filter
from scipy.spatial import ConvexHull, Delaunay
from skimage import feature
from skimage.measure import regionprops, find_contours
from skimage.draw import polygon
from skimage.morphology import remove_small_objects, binary_closing, ball #, binary_dilation
from skimage import segmentation
from skimage.segmentation import find_boundaries
# import math
import napari
from collections import defaultdict
import time
from typing import DefaultDict, List, Tuple

if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    __package__ = 'ImgProc'
from . import directional_analysis
from . import labelEditor

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

def fill_radial_cracks(mask3d, radius=2):
    new_mask = np.zeros_like(mask3d)
    label_ids = np.unique(mask3d)
    label_ids = label_ids[label_ids!=0]

    new_label = 1
    for id in label_ids:
        single_cell = (mask3d==id)
        closed = binary_closing(single_cell, ball(radius))
        filled = binary_fill_holes(closed)
        new_mask [filled] = new_label
        new_label += 1
    return new_mask

def convex_hull_Id(label_mask,label_id:int):
    coords = np.column_stack(np.where(label_mask == label_id))
    if len(coords) < 4: #less than 4 points no Canvex hull
        return label_mask == label_id
    hull = ConvexHull(coords)
    delaunary = Delaunay(coords[hull.vertices])
    min_z, min_y, min_x = coords.min(axis=0)
    max_z, max_y, max_x = coords.max(axis=0)

    zz, yy, xx = np.mgrid[
                 min_z:max_z+1,
                 min_y:max_y+1,
                 min_x:max_x+1]
    grid_points = np.stack((zz, yy, xx), axis=-1).reshape(-1,3)
    # filter out the points in the mask: they must be in the convex hull
    grid_flat_index = np.ravel_multi_index(grid_points.T, label_mask.shape)
    coords_flat_index = np.ravel_multi_index(coords.T, label_mask.shape)
    mask_not_origin = ~np.isin(grid_flat_index, coords_flat_index)
    query_points = grid_points[mask_not_origin]
    inside = delaunary.find_simplex(query_points) >= 0
    fill_coords = query_points[inside]
    final_coords = np.vstack([coords, fill_coords])
    solid_mask = np.zeros_like(label_mask,dtype=bool)
    solid_mask[final_coords[:,0],final_coords[:,1],final_coords[:,2]] = True
    return solid_mask

def convex_hull(label_mask):
    label_mask = np.asarray(label_mask)
    slices = find_objects(label_mask)
    out = np.zeros_like(label_mask,dtype = label_mask.dtype)
    for label_id, slc in enumerate(slices,start=1):
        if slc is None: continue
        submask = label_mask[slc]
        sub_convexhull = convex_hull_Id(submask,label_id)
        out[slc] = np.where(sub_convexhull,label_id,out[slc])
    return out

class UnionFind: # this algo is implement by disjoint set
    def __init__(self, size):
        self.parents = list(range(size))
    def find(self, x:int)->int:
        if self.parents[x] != x: self.parents[x] = self.find(self.parents[x])
        return self.parents[x]
    def union(self, x:int, y:int)-> None:
        px, py = self.find(x), self.find(y)
        if px != py: self.parents[py] = px

# a function to merge single cell split into multiple labels: with contact voxel number and centroid distance
def merge_split_label(mask, centroid_thresh:int = 100, contact_thresh:int = 5000):
    mask = mask.copy() # allocate new memory (to prevent contamination from ravel)
    props = regionprops(mask)
    centroid = np.array([p.centroid for p in props])
    labels = np.array([p.label for p in props])
    contact_counter = defaultdict(int)

    # analyse voxel contact
    # voxel box (3x3x3) scanning algor + boundary
    start = time.time()
    is_boundary = find_boundaries(mask,mode='inner')
    for z in range(1,mask.shape[0]-1):
        for y in range(1,mask.shape[1]-1):
            for x in range(1,mask.shape[2]-1):
                if not is_boundary[z, y, x]: continue
                # if mask[z,y,x] == 0: continue
                center = mask[z,y,x]
                neighbors = mask[z-1:z+2,y-1:y+2,x-1:x+2].ravel()
                for n in neighbors:
                    if n==0 or n==center: continue
                    key = tuple(sorted((center,n)))
                    contact_counter[key] += 1
    print("time spend on 333 voxel scan", time.time() - start)
    '''
    # dilation algorithm: slower than 333 voxel scan + boundary
    start = time.time()
    structure = np.ones((3,3,3),dtype=bool)
    labels = np.array([p.label for p in props if p.label != 0])
    for lb in labels:
        region = (mask == lb)
        dilated = binary_dilation(region, structure=structure )
        # overlap = mask[np.logical_and(dilated,~region)]
        overlap = mask[dilated]
    overlap_ids, counts = np.unique(overlap,return_counts=True)
    for oid, cnt in zip(overlap_ids, counts):
        if oid == 0 or oid == lb: continue
        contact_counter[tuple(sorted((lb,oid)))] += cnt
    print("time spend on building contact", time.time() - start)
    '''
    start = time.time()
    groups: DefaultDict[int, List[int]] = defaultdict(list) # store the groups
    # fa = np.zeros((labels.max()+1,1),dtype=int) # to store the father (disjoint-set)
    labels_len = len(labels)
    uf = UnionFind(labels.max()+1)
    for i in range(labels_len):
        c1 = centroid[i]
        for j in range(i+1,labels_len):
            c2 = centroid[j]
            dist = np.linalg.norm(np.array(c1)-np.array(c2))
            contact_key = tuple(sorted((labels[i],labels[j])))
            if dist < centroid_thresh and contact_counter[contact_key] >= contact_thresh:
                uf.union(contact_key[0], contact_key[1])
                # generate a union from contact_key[0] to contact_key[1]
                # print(contact_key)
    for lb in labels: groups[uf.find(lb)].append(lb)
    # print(groups)
    print("time spend on dj set", time.time() - start)
    new_label = 1
    new_mask = np.zeros_like(mask)
    for k in groups:
        for lb in groups[k]:
            new_mask[mask == lb] = new_label
        new_label += 1
    return new_mask

def filter_small(label_mask, vol_thresh:int = 1000):
    props = regionprops(label_mask)
    new_mask = np.zeros_like(label_mask)
    new_label = 1
    for prop in props:
        if prop.area > vol_thresh:
            new_mask[label_mask == prop.label] = new_label
            new_label += 1
    return new_mask

def delete_bad(label_mask, bad_label):
    if len(bad_label) == 0: return label_mask.copy()
    labels = np.unique(label_mask)
    labels = labels[labels!=0]
    label_mask = label_mask.copy()
    new_label = 1
    for lb in bad_label:
        label_mask[label_mask == lb] = 0
    for lb in labels:
        if lb in bad_label: continue
        else:
            label_mask[label_mask == lb] = new_label
            new_label += 1
    return label_mask

def ellipsoid(rx,ry,rz):
    z,y,x = np.ogrid[-rz:rz+1, -ry:ry+1, -rx:rx+1]
    mask = (x/rx)**2 + (y/ry)**2 + (z/rz)**2
    return mask.astype(np.uint8)

def watershed_mask(label_mask):
    print(f'shape label {label_mask.shape}')
    new_labels = np.zeros_like(label_mask)
    if True:
        label_area = label_mask.copy()
        distance = distance_transform_edt(label_area,sampling=(1.5,0.08,0.08))
        distance_smooth = gaussian_filter(distance,sigma=(100/18.75/3,100/3,100/3))
        # print(f'shape distance {distance.shape}, shape label {label_area.shape}')
        coords = feature.peak_local_max(distance_smooth, labels=label_area,min_distance=6)
        # print(coords)
        markers = np.zeros_like(label_area,dtype=int)
        for i, (z,r,c) in enumerate(coords,start=1):
            markers[z,r,c] = i
        new_labels = segmentation.watershed(-distance_smooth,markers,mask=label_area)
    return new_labels

def relabel(label_mask):
    # print(label_mask)
    labels = np.unique(label_mask)
    # print(labels)
    labels = labels[labels!=0]
    labels.sort()
    label_mask = label_mask.copy()
    fill = -1
    for lb in range(1,len(labels)+1):
        if lb not in labels:
            label_mask[label_mask==labels[fill]] = lb
            fill -= 1
    return label_mask

def install_manual_merge_remove(viewer, label_layer, label_mask, save):
    merge_queue = []
    @label_layer.mouse_drag_callbacks.append
    def remove_labels(layer,event):
        print(f'position: {event.position}')
        nonlocal label_mask
        coord = tuple(np.round(layer.world_to_data(event.position)).astype(int))
        label_id = layer.data[coord]
        print(f'coord = {coord}, label = {label_id}')

        if event.button == 2: # right click
            print('Right click: remove label')
            if label_id > 0:
                print(f'Remove label {label_id}')
                layer.data[layer.data == label_id] = 0
                label_mask[label_mask == label_id] = 0
                layer.refresh()
        elif event.button == 1: # left click
            print(f'Left click: merge labels {merge_queue}')
            if label_id > 0: merge_queue.append(label_id)
        else: print(f'Click == {event.button}'); return
    # label_layer.mouse_drag_callbacks.append(remove_labels)
    @viewer.bind_key('Shift-m')
    def merge(v):
        nonlocal merge_queue
        nonlocal label_mask
        nonlocal label_layer
        print(f'merge queue: {merge_queue}')
        if len(merge_queue) > 1:
            new_label = min(merge_queue)
            for id in merge_queue:
                if id == new_label: continue
                label_mask[label_mask == id] = new_label
                label_layer.data[label_layer.data == id] = new_label
            label_layer.refresh()
        merge_queue.clear()
    @viewer.bind_key('c')
    def clear_queue(v):
        nonlocal merge_queue
        print(f'clear queue: {merge_queue}')
        merge_queue.clear()
    @viewer.bind_key('q')
    def finish(v):
        print('Finished editing')
        v.close()

    @viewer.bind_key('e')
    def exit_no_save(v):
        print('Return without save')
        v.close()
        nonlocal save
        save = False
        return
    @viewer.bind_key('s')
    def exit_with_save(v):
        print('Return with save')
        v.close()
        nonlocal save
        save = True

def call_postProc(path:str|None = None, subfolder:str|None = None, filename:str|None = None, napari_view:bool|None = None, save=True):
                        # To do: Turn this into a callable function for CLI add another main function to call this function ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # Come up with a better merging algorithm ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # Have a cellpose flow viewer (Turn 3D coord into RGB?) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # Binary close the mask ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if path is None: print('path is None'); return
    if not os.path.exists(path): print(f"path {path} not exists");return
    if filename is None: print('filename is None'); return
    # if manual is None: manual = True
    if napari_view is None: napari_view = True # the switch to turn on / off napari
    if save is None: save = True
    print(filename)
    print(os.getcwd(),path)

    with tifffile.TiffFile(os.path.join(path, filename+'.tif')) as tifimg:
        img = tifimg.asarray()

    masks = np.load(os.path.join(path, subfolder, filename + '_seg.npy'), allow_pickle=True) # dict_keys(['outlines', 'masks', 'chan_choose', 'ismanual', 'filename', 'flows', 'diameter'])
    # print(masks.item().keys()) # dict_keys(['outlines', 'masks', 'chan_choose', 'ismanual', 'filename', 'flows', 'diameter'])
    mask = masks.item()['masks']
    #flows = masks.item()['flows']
    #print(flows[0].shape,flows[1].shape,flows[2].shape,len(flows[3]),flows[4].shape)

    print("filter small")
    filtered_small_masks = filter_small(mask,10000)

    if napari_view:
        viewer = napari.Viewer()
        viewer.add_image(img, name = 'Raw Image', colormap = 'gray', contrast_limits = (np.percentile(img,1), np.percentile(img,99)))
        viewer.add_labels(label_boundary_fast(mask),name = '3D cell masks')
        label_layer = viewer.add_labels(label_boundary_fast(filtered_small_masks),name = '3D cell masks small filtered')
        
        # label_layer = viewer.add_labels(label_boundary_fast(merged_label),name = '3D cell masks small filtered fused')

        #install_manual_merge_remove(viewer, label_layer, manual_filtered_mask, save)
        editor = labelEditor.LabelEditor()
        editor.install(viewer=viewer, label_layer=label_layer,label_mask=filtered_small_masks, save=save)
        manual_filtered_mask, save, _ = editor.run()
        
        # print("Click on bad labels")
        # napari.run()

        # Sort labels
        manual_filtered_mask = relabel(manual_filtered_mask)

        # viewer = napari.Viewer()
        # viewer.add_image(img, name='Raw Image', colormap='gray',
        #                         contrast_limits=(np.percentile(img, 1), np.percentile(img, 99)))
        # viewer.add_labels(label_boundary_fast(manual_filtered_mask), name='manually filtered mask')
        # napari.run()
    print(f'save = {save}')
    if save:
        np.savez(path+filename+"_masks.npz",
            original = mask,
            filtered = filtered_small_masks,
            # merged = merged_label,
            manual_filtered = manual_filtered_mask,
            )
    return manual_filtered_mask

def three_masks(path = None, path_cytomask = None,prefix = None, channel = None):
    # to get 3 masks: labelled nuclei mask (cellpose + merge + manual picked), binary nuclei mask (cellpose + view + binary closing + binary), binary cytoplasm mask (cytomask + binary)

    if prefix is None: prefix = ''
    exclude_on_edge = 'exclude_on_edge/'
    cellposemask = './'
    filename_nuclei = prefix + '_C' + str(channel[0])
    filename_cytoplasm = prefix + '_C' + str(channel[1]) + '.tif'
    to_save = {}
    # if path is not None: to_save['labelled'] = call_postProc(path,exclude_on_edge,filename_nuclei,save = False)
    if path is not None: to_save['labelled_exclude'] = call_postProc(path,cellposemask,filename_nuclei,save = False)
    if path is not None: binary_nuclei = call_postProc(path,cellposemask,filename_nuclei,save=False); to_save['labelled_all'] = binary_nuclei; to_save['binary'] = (binary_nuclei != 0)
    if path_cytomask is not None: to_save['cytomask'] = directional_analysis.call_cyto_mask_fiber(path=path_cytomask, filename=filename_cytoplasm, eps=1e-4)
    np.savez(os.path.join(path, prefix + '_4masks.npz'), **to_save)


def batch_call_three_masks(path, path_cytomask, channel):
    filelist = os.listdir(path)
    filelist = [f for f in filelist if f.find('C0')!=-1]
    for f in filelist:
        print(f, f.split('_C0')[0])
        three_masks(path=path, path_cytomask=path_cytomask, prefix=f.split('_C0')[0], channel = channel)
        # three_masks(path=path, path_cytomask=path_cytomask, prefix=f.split('_C')[0], channel = channel)

if __name__ == '__main__':
    path_prefix = '/mnt/box/HyperOsmo/20251017_Exposure_Recov_SINE48_ki67_H3K27me3/tiff/'
    file_prefix = '2025-10-11_10AWT_48_Hypo50_01_63x_004'
    # call_postProc(path_prefix,'.',filename=file_prefix,save=False)
    # exit()
    channel = (3,1) # (nuclear channel, actin channel) _C?.
    # 2025-10-02_10AWT_05_Hypo50_01_63x_004_C3
    # 
    # three_masks(path=path_prefix, path_cytomask=path_prefix, prefix=file_prefix, channel = channel)
    batch_call_three_masks(path=path_prefix, path_cytomask=path_prefix, channel = channel)

'''


if 0: # test call_postProc function -> labelEditor package function
    path = '/mnt/SammyRis/Sammy/3D_hoechst_250731_Day1/'
    subfolder = 'exclude_on_edge/'
    filename = '10a_5kPa_100Pa_CMO_Hoechst_exposure-01(1)_T0_C1'
    call_postProc(path, subfolder, filename, save=False)

'''