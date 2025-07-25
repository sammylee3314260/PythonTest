import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tifffile
from dask.array import outer
from dask.array.reshape import contract_tuple
from scipy.ndimage import binary_fill_holes, find_objects, binary_dilation
from scipy.spatial import ConvexHull, Delaunay
from skimage.measure import regionprops, find_contours
from skimage.draw import polygon
from skimage.morphology import remove_small_objects, binary_closing, ball #, binary_dilation
from skimage.segmentation import find_boundaries
import math
import napari
from collections import defaultdict
import time
from typing import DefaultDict, List, Tuple

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
    new_mask = np.zeros_like(mask)
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

if __name__ =='__main__':
    napari_view = 1 # the switch to turn on / off napari
    path = '/mnt/SammyRis/Sammy/2025072021_exp_recov_3D/mask_ani_5_diam_150/'
    filename = '2025-07-20_10AWT_Ctrl_02_63x_001_C3'
    print(filename)
    print(os.getcwd(),path)

    if not os.path.exists(path):print(f"path {path} not exists");exit()

    with tifffile.TiffFile(path+filename+'.tif') as tifimg:
        img = tifimg.asarray()

    if napari_view: viewer = napari.Viewer()
    if napari_view: viewer.add_image(img, name = 'Raw Image', colormap = 'gray', contrast_limits = (np.percentile(img,1), np.percentile(img,99)))

    masks = np.load(path+filename+'_seg.npy', allow_pickle=True) # dict_keys(['outlines', 'masks', 'chan_choose', 'ismanual', 'filename', 'flows', 'diameter'])
    print(masks.item().keys())
    mask = masks.item()['masks']
    #flows = masks.item()['flows']
    #print(flows[0].shape,flows[1].shape,flows[2].shape,len(flows[3]),flows[4].shape)
    if napari_view: viewer.add_labels(label_boundary_fast(mask),name = '3D cell masks')
    #print("ndim of mask = ",mask.shape)

    print("filter small")
    filtered_small_masks = filter_small(mask,10000)
    if napari_view: viewer.add_labels(label_boundary_fast(filtered_small_masks),name = '3D cell masks small filtered')

    print("merge split")
    merged_label = merge_split_label(filtered_small_masks,contact_thresh=10000)
    if napari_view: viewer.add_labels(label_boundary_fast(merged_label),name = '3D cell masks small filtered fused')

    print("convex hull")
    convex_hull_mask = convex_hull(merged_label)
    if napari_view: viewer.add_labels(label_boundary_fast(convex_hull_mask),name = '3D cell masks small filtered fused convex hull')

    if napari_view:
        print("Enter bad labels, part with blank:")
        napari.run()
        bad_labels = list(map(int, input().split()))
        print(bad_labels)
        manual_filtered_mask = delete_bad(merged_label,bad_labels)
        manual_filtered_convex_hull_mask = delete_bad(convex_hull_mask,bad_labels)
        viewer = napari.Viewer()
        viewer.add_image(img, name='Raw Image', colormap='gray',
                                contrast_limits=(np.percentile(img, 1), np.percentile(img, 99)))
        viewer.add_labels(label_boundary_fast(manual_filtered_mask), name='manually filtered mask')
        viewer.add_labels(label_boundary_fast(manual_filtered_convex_hull_mask), name = 'manually filtered convHull')
        napari.run()
    np.savez(path+filename+"_masks.npz",
            original = mask,
            filtered = filtered_small_masks,
            merged = merged_label,
            convex_hull = convex_hull_mask,
            manual_filtered = manual_filtered_mask,
            manual_filtered_convex_hull = manual_filtered_convex_hull_mask)
    exit()

'''
masks = np.load(path+filename, allow_pickle=True)
print(masks.item()['filename']) # dict_keys(['outlines', 'masks', 'chan_choose', 'ismanual', 'filename', 'flows', 'diameter'])
print(type(masks.item()['masks']))

cleaned = remove_small_objects(masks.item()['masks'], min_size=1000)

exit()
contours = measure.find_contours(cleaned)
fig, ax = plt.subplots()
ax.imshow(img,cmap='gray')
for contour in contours:
    ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='cyan')
plt.tight_layout()
plt.show()

roi_bool = cleaned.astype(bool)
img_roi = img[roi_bool]
total_intensity = np.sum(img_roi)
mean_intensity = np.mean(img_roi)
count_roi = np.sum(roi_bool)
std_intensity = np.std(img_roi)
sem_intensity = std_intensity / math.sqrt(count_roi)
#for f in files:
'''