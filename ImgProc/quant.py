import tifffile
import numpy as np
import cvxpy as cp
import os
from skimage.measure import regionprops
from scipy.spatial import ConvexHull

def compute_mvee(label_masks, tol = 1e-5):
    coords = np.argwhere(label_masks)
    hull = ConvexHull(coords)
    points = coords[hull.vertices]
    N,d = points.shape # points is a np.array of coordinates
    Q = np.column_stack((points,np.ones(N))) # add one dimention
    P = cp.Variable((d+1,d+1),PSD=True)
    constraints = [cp.sum(cp.multiply(Q@P,Q),axis=1) <= 1]
    prob = cp.Problem(cp.minimize(-cp.log_det(P)),constraints)
    prob.solve(abstol=tol,reltol=tol)
    P_val = P.value
    A = P_val[:d,:d]
    c = -np.linalg.solve(A,P_val[:d,-1])
    return A, c

if __name__ == '__main__':
    path = ""
    file = ""
    if not os.path.exists(path): print(f"Path {path} not exists.");exit()
    with tifffile.Tifffile(path + file + '.tif') as tiffimg:
        img = tiffimg.asarray()
    label_masks = np.load(path+file+'_masks.npy')
    props = regionprops(label_image = label_masks, intensity_image = img)

    # if I want to use pandas
    # props_table = regionprops_table(label_image = label_masks, intensity_image = img)
    # tables = pd.DataFrame(props_table)

    # some properties: prop.area, area_convex, area_filled,
    # eccentricity, intensity_mean, intensity_std, solidity

    # If I have these, I only have to make sure the mask is good,
    # I can get label, volume (area), intensity(mean), eccentricity, and solidity.