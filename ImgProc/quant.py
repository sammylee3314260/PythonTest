import tifffile
import numpy as np
import cvxpy as cp
import pandas as pd
import os
import re
from skimage.measure import regionprops_table, regionprops
from scipy.spatial import ConvexHull
from scipy.ndimage import zoom
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    __package__ = 'ImgProc'
from . import directional_analysis
from . import image_display

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

def cyto_quant(image, mask,properties = None):
    # this is a function to quant 2D image
    if not properties: return regionprops_table(label_image=mask,intensity_image=image)
    return regionprops_table(label_image=mask,intensity_image=image,properties=properties)

def call_cyto_quant(path = "", img_filter="", mask_filter="",pattern="",mask_name="",sigma = 2,eps = 1e-4,display=False,cellpose = False,concat=[]):
    if not os.path.exists(path): print(f"path {path} not exists");return
    filelist = os.listdir(path)
    imglist = [f for f in filelist if f.find('.tif')!=-1]
    if img_filter != "": imglist = [f for f in imglist if f.find(img_filter)!=-1]
    print(imglist)
    # masklist = [f for f in filelist if f.find('cytomask')!=-1]
    all_data = []

    for f in imglist:
        name, _ = os.path.splitext(f)
        
        match = re.match(pattern,name)
        print(name)
        if not match: print('match is none');continue

    # '10A_5kPa_Ctrl_POS48h_40xoil_max_C2.tif'
    # '10A_5kPa_Ctrl_POS48h_40xoil_max_C1_cytomask.npy'
        maskfile = '_'.join(list(match.groups()[0:-1])+concat)

        if not os.path.exists(os.path.join(path,maskfile)): print(f"file {os.path.join(path,maskfile)} not exists");continue
        with tifffile.TiffFile(os.path.join(path,f)) as tifimg:
            img = tifimg.asarray()
        if cellpose:
            mask = np.load(os.path.join(path,maskfile),allow_pickle=True)
            mask = mask.item()['masks']
            mask = np.where(mask,1,0)
        else:
            mask = np.load(os.path.join(path,maskfile))
            if mask_filter.endswith('.npz'): mask = mask[mask_name]
        if mask.dtype == 'bool':
            mask = np.where(mask,1,0)
        # properties = ['label', 'area', 'intensity_mean']
        properties = ['label', 'area', 'area_convex', 'area_filled','intensity_max', 'intensity_mean','intensity_min', 'intensity_std', 'solidity']
        df = pd.DataFrame(cyto_quant(img, mask, properties))
        
        
        if True: # directional analysis
            theta , coherence, energy = directional_analysis.compute_orientation_tensor_2d(img, sigma=sigma, eps=eps)
            if display: image_display.analy_display(img,theta,coherence,energy)
            coherence_nan = np.where(mask,coherence,np.nan)
            df['coherence_avg'] = np.nanmean(coherence_nan); df['coherence_median'] = np.nanmedian(coherence_nan)
        # 2025-07-21_10AWT_Hyper969_02_63x_002_max_C1_cytomask.npy
        # pattern = r'(?P<prefix>.+?)_(?P<type>\w+?)_(?P<group>\w+?)_(?P<slnum>\d+?)_(?P<magni>\w+?)_(?P<scnum>\d+?)_max_(?P<channel>\w+)'
        df['Date'] = match.group('prefix')
        df['Group'] = match.group('group')
        # df['Time'] = match.group('time')
        df['Image'] = name;df['Mask'] = os.path.splitext(maskfile)[0]
        all_data.append(df)
    
    result_df = pd.concat(all_data, ignore_index=True)
    print(result_df)
    custom_order = ['Hyper969','Hyper484','Ctrl','Hypo66','Hypo50','Hypo33']
    cat_type = pd.api.types.CategoricalDtype(categories=custom_order,ordered=True)
    result_df['Group'] = result_df['Group'].astype(cat_type)
    sorted_df = result_df.sort_values('Group')
    print(sorted_df)
    return sorted_df, custom_order


def quant_nucleus_physical(masks):
    # quant_nucleus_physical: this function analyse the physical properties of a labelled nucleus mask
    # properties=['label', 'area', 'area_convex', 'area_filled', 'eccentricity','solidity']
    return regionprops_table(label_image=masks,properties=['label', 'area', 'area_convex', 'area_filled','axis_major_length','axis_minor_length',
                                                            'bbox','solidity'])

def call_nuc_phys(path = "", mask_filter="",pattern="",mask_name="",spacing=None):
    print("quant_nucleus_physical, make sure the nucleus mask is exclude_on_edge")
    filelist = os.listdir(path)
    filelist = [f for f in filelist if f.find(mask_filter)!=-1]
    all_data = []
    for f in filelist:
        print(f)
        name, ext = os.path.splitext(f)
        match = re.match(pattern, name)
        if match is None:
            print(f"No match found: name {name}, pattern:{match.regs}")
            continue
        mask = np.load(os.path.join(path,f))
        if ext == '.npz': mask = mask[mask_name]
        #zoom_factors = list(spacing)
        #mask = zoom(mask.astype(float),zoom=zoom_factors, order=0)
        #mask = mask.astype(np.uint8)
        df = pd.DataFrame(quant_nucleus_physical(mask))
        df['Slide num'] = match.group('slnum'); df['Date'] = match.group('prefix'); df['Group'] = match.group('group');df['Mask'] = name
        all_data.append(df)
    result_df = pd.concat(all_data, ignore_index=True)
    date_order = ['2025-07-20','2025-07-21']
    result_df['Date'] = pd.Categorical(result_df['Date'],categories=date_order,ordered=True)
    custom_order = ['Hyper969','Hyper484','Ctrl','Hypo66','Hypo50','Hypo33']
    result_df['Group'] = pd.Categorical(result_df['Group'],categories=custom_order,ordered=True)
    sorted_df = result_df.sort_values(['Date','Group'])
    print(sorted_df)
    return sorted_df, custom_order


if __name__ == '__main__':
    path = '/mnt/SammyRis/Sammy/YAP_Actin_lamAC_max_proj/'
    filename = 'YAP_Actin_lamAC_nuclear_LamAC'
    mask_name='manual_filtered'
    img_filter = 'C3.tif' # C1:actin, C2: vimentin C0: Ki67, C3:hoechst
    mask_filter = 'C3_seg.npy'
    # 2025-07-21_10AWT_Ctrl_01_63x_001_max_C2.tif
    # 2025-07-21_10AWT_Hyper969_02_63x_002_max_C1_cytomask.npy
    pattern = r'(?P<prefix>.+?)_(?P<type>\w+?)_(?P<group>\w+?)_(?P<slnum>\d+?)_(?P<magni>\w+?)_(?P<scnum>\d+?)_max_(?P<channel>\w+)'
    #I can directly use split('_') for this instead of regex to generalize the package

    if 1: # new 3D nucleus analysis
        if not os.path.exists(path): print(f"Path {path} not exists.");exit()
        df, order = call_cyto_quant(path,img_filter,mask_filter,pattern,cellpose=True,mask_name=mask_name,concat=['max',mask_filter])
        df.to_pickle(os.path.join(path,filename+".pkl"))
        df.to_csv(os.path.join(path, filename+".csv"),index=False)
    elif 0: # new 2D cytoplasm analysis
        if not os.path.exists(path): print(f"Path {path} not exists.");exit()
        df, order = call_cyto_quant(path,img_filter,mask_filter,pattern,sigma=2,eps=2e-4,cellpose=False,concat=['max',mask_filter])
        df.to_pickle(os.path.join(path,filename+".pkl"))
        df.to_csv(os.path.join(path, filename+".csv"),index=False)
    elif 1: # read from saved
        file = filename+'.pkl'
        df = pd.read_pickle(os.path.join(path,file))
        order = df['Group'].drop_duplicates().tolist()
    elif 0: # analyse nuclear physical properties
        df, order = call_nuc_phys(path,mask_filter,pattern,mask_name)
        df.to_pickle(os.path.join(path,filename+".pkl"))
        df.to_csv(os.path.join(path, filename+".csv"),index=False)
    else: exit()
    
    fignum = 0
    df_ori = df
    order.append('All')
    order = ['Hyper969 POS24h','Hyper484 POS24h','Ctrl POS24h','Hypo66 POS24h','Hypo50 POS24h','Hypo33 POS24h','Ctrl POS48h','Hypo66 POS48h','Hypo50 POS48h','Hypo33 POS48h','All']
    colors = sns.color_palette("dark:gray",n_colors=len(order)+2)[0:len(order)]
    palette = dict(zip(order,colors))

    plot_data = 'intensity_mean'
    plot_data = 'coherence_avg'
    df['GroupTime'] = df['Group'].astype(str) + ' ' + df['Time']
    df[plot_data] = df['area'] * df['intensity_mean']
    df_all = df.copy()
    df_all['GroupTime'] = 'All'
    df_combined = pd.concat([df,df_all],ignore_index=True)
    df = df_combined
    
    sns.barplot(data=df,x='GroupTime',y=plot_data,order=order,estimator=np.mean,palette=palette, errorbar='se',capsize=0.2)
    sns.stripplot(data=df,x='GroupTime',y=plot_data,order=order)
    plt.show()
    exit()

    fignum += 1; plt.figure(fignum)
    fig, axes = plt.subplots(2,1,num=fignum)
    fig.suptitle("Vimentin Coherency",weight='bold')
    # sns.barplot(data=df[df['Date']=='2025-07-20'],x='Group',y=plot_data,order=order,estimator=np.mean,palette=palette, errorbar='se',capsize=0.2,ax=axes[0])
    axesnum = 0
    sns.violinplot(data=df[df['Date']=='2025-07-20'],x='Group',y=plot_data,order=order,palette=palette,ax=axes[axesnum])
    sns.stripplot(data=df[df['Date']=='2025-07-20'],x='Group',y=plot_data,order=order,ax=axes[axesnum])
    axes[axesnum].set_title("Exposure 24 hr",weight='bold')
    axesnum = 1
    # sns.barplot(data=df[df['Date']=='2025-07-21'],x='Group',y=plot_data,order=order,estimator=np.mean,palette=palette, errorbar='se',capsize=0.2,ax=axes[1])
    sns.violinplot(data=df[df['Date']=='2025-07-21'],x='Group',y=plot_data,order=order,palette=palette,ax=axes[axesnum])
    sns.stripplot(data=df[df['Date']=='2025-07-21'],x='Group',y=plot_data,order=order,ax=axes[axesnum])
    axes[axesnum].set_title("Recover 24 hr",weight='bold')
    for ax in axes.flat:
        ax.set_ylabel("Coherency mean",weight='bold')
        ax.set_xlabel(None)
        ax.set_ylim(-0.25,1.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.tick_params(axis='both',which='major',width=2,length = 6)
    fig.tight_layout()
    plt.show()
    exit()

    '''
    # plot bbox 3 dimention
    df = df_ori
    df['Z'] = (df['bbox-3'] - df['bbox-0'])
    df['Y'] = (df['bbox-4'] - df['bbox-1'])
    df['X'] = (df['bbox-5'] - df['bbox-2'])
    df_all=df.copy()
    df_all['Group'] = 'All'
    df_combined = pd.concat([df,df_all],ignore_index=True)
    df = df_combined
    fignum += 1; plt.figure(fignum)
    fig, axes = plt.subplots(3,1,num=fignum)
    fig.suptitle("Exposure 24 hr",weight='bold')
    sns.barplot(data=df[df['Date']=='2025-07-20'],x='Group',y='Z',order=order,estimator=np.mean,palette=palette, errorbar='se',capsize=0.2,ax=axes[0])
    sns.stripplot(data=df[df['Date']=='2025-07-20'],x='Group',y='Z',order=order,ax=axes[0])
    axesnum = 0
    axes[axesnum].set_title("Nuclear Z height",weight='bold')
    axes[axesnum].set_ylabel("Length (um)",weight='bold')
    axes[axesnum].set_xlabel(None)
    axes[axesnum].spines['top'].set_visible(False)
    axes[axesnum].spines['right'].set_visible(False)
    axes[axesnum].spines['left'].set_linewidth(2)
    axes[axesnum].spines['bottom'].set_linewidth(2)
    
    sns.barplot(data=df[df['Date']=='2025-07-20'],x='Group',y='Y',order=order,estimator=np.mean,palette=palette, errorbar='se',capsize=0.2,ax=axes[1])
    sns.stripplot(data=df[df['Date']=='2025-07-20'],x='Group',y='Y',order=order,ax=axes[1])
    axesnum = 1
    axes[axesnum].set_title("Nuclear y width",weight='bold')
    axes[axesnum].set_ylabel("Length (um)",weight='bold')
    axes[axesnum].set_xlabel(None)
    axes[axesnum].spines['top'].set_visible(False)
    axes[axesnum].spines['right'].set_visible(False)
    axes[axesnum].spines['left'].set_linewidth(2)
    axes[axesnum].spines['bottom'].set_linewidth(2)
    axes[axesnum].tick_params(axis='both',which='major',width=2,length = 6)
    sns.barplot(data=df[df['Date']=='2025-07-20'],x='Group',y='X',order=order,estimator=np.mean,palette=palette, errorbar='se',capsize=0.2,ax=axes[2])
    sns.stripplot(data=df[df['Date']=='2025-07-20'],x='Group',y='X',order=order,ax=axes[2])
    axesnum = 2
    axes[axesnum].set_title("Nuclear X width",weight='bold')
    axes[axesnum].set_ylabel("Length (um)",weight='bold')
    axes[axesnum].set_xlabel(None)
    axes[axesnum].spines['top'].set_visible(False)
    axes[axesnum].spines['right'].set_visible(False)
    axes[axesnum].spines['left'].set_linewidth(2)
    axes[axesnum].spines['bottom'].set_linewidth(2)
    axes[axesnum].tick_params(axis='both',which='major',width=2,length = 6)
    fig.tight_layout()
    plt.show(block=False)

    fignum += 1; plt.figure(fignum)
    fig, axes = plt.subplots(3,1,num=fignum)
    fig.suptitle("Recover 24 hr",weight='bold')
    sns.barplot(data=df[df['Date']=='2025-07-21'],x='Group',y='Z',order=order,estimator=np.mean,palette=palette, errorbar='se',capsize=0.2,ax=axes[0])
    sns.stripplot(data=df[df['Date']=='2025-07-21'],x='Group',y='Z',order=order,ax=axes[0])
    axesnum = 0
    axes[axesnum].set_title("Nuclear Z height",weight='bold')
    axes[axesnum].set_ylabel("Length (um)",weight='bold')
    axes[axesnum].set_xlabel(None)
    axes[axesnum].spines['top'].set_visible(False)
    axes[axesnum].spines['right'].set_visible(False)
    axes[axesnum].spines['left'].set_linewidth(2)
    axes[axesnum].spines['bottom'].set_linewidth(2)
    axes[axesnum].tick_params(axis='both',which='major',width=2,length = 6)
    sns.barplot(data=df[df['Date']=='2025-07-21'],x='Group',y='Y',order=order,estimator=np.mean,palette=palette, errorbar='se',capsize=0.2,ax=axes[1])
    sns.stripplot(data=df[df['Date']=='2025-07-21'],x='Group',y='Y',order=order,ax=axes[1])
    axesnum = 1
    axes[axesnum].set_title("Nuclear Y width",weight='bold')
    axes[axesnum].set_ylabel("Length (um)",weight='bold')
    axes[axesnum].set_xlabel(None)
    axes[axesnum].spines['top'].set_visible(False)
    axes[axesnum].spines['right'].set_visible(False)
    axes[axesnum].spines['left'].set_linewidth(2)
    axes[axesnum].spines['bottom'].set_linewidth(2)
    axes[axesnum].tick_params(axis='both',which='major',width=2,length = 6)
    sns.barplot(data=df[df['Date']=='2025-07-21'],x='Group',y='X',order=order,estimator=np.mean,palette=palette, errorbar='se',capsize=0.2,ax=axes[2])
    sns.stripplot(data=df[df['Date']=='2025-07-21'],x='Group',y='X',order=order,ax=axes[2])
    axesnum = 2
    axes[axesnum].set_title("Nuclear X width",weight='bold')
    axes[axesnum].set_ylabel("Length (um)",weight='bold')
    axes[axesnum].set_xlabel(None)
    axes[axesnum].spines['top'].set_visible(False)
    axes[axesnum].spines['right'].set_visible(False)
    axes[axesnum].spines['left'].set_linewidth(2)
    axes[axesnum].spines['bottom'].set_linewidth(2)
    axes[axesnum].tick_params(axis='both',which='major',width=2,length = 6)
    fig.tight_layout()
    plt.show()
    exit()'''
    

    df = df_ori
    df_all=df.copy()
    df_all['Group'] = 'All'
    df_combined = pd.concat([df,df_all],ignore_index=True)
    df = df_combined
    fignum += 1; plt.figure(fignum)
    fig, axes = plt.subplots(2,1,num=fignum)
    fig.suptitle("Nuclear Solidity",weight='bold')
    sns.barplot(data=df[df['Date']=='2025-07-20'],x='Group',y='solidity',order=order,estimator=np.mean,palette=palette, errorbar='se',capsize=0.2,ax=axes[0])
    sns.stripplot(data=df[df['Date']=='2025-07-20'],x='Group',y='solidity',order=order,ax=axes[0])
    axesnum = 0
    axes[axesnum].set_title("Exposure 24 hr",weight='bold')
    axes[axesnum].set_ylim(0.6,1)
    axes[axesnum].set_ylabel("Solidity",weight='bold')
    axes[axesnum].set_xlabel(None)
    axes[axesnum].spines['top'].set_visible(False)
    axes[axesnum].spines['right'].set_visible(False)
    axes[axesnum].spines['left'].set_linewidth(2)
    axes[axesnum].spines['bottom'].set_linewidth(2)
    axes[axesnum].tick_params(axis='both',which='major',width=2,length = 6)

    sns.barplot(data=df[df['Date']=='2025-07-21'],x='Group',y='solidity',order=order,estimator=np.mean,palette=palette, errorbar='se',capsize=0.2,ax=axes[1])
    sns.stripplot(data=df[df['Date']=='2025-07-21'],x='Group',y='solidity',order=order,ax=axes[1])
    axesnum = 1
    axes[axesnum].set_title("Recover 24 hr",weight='bold')
    axes[axesnum].set_ylim(0.6,1)
    axes[axesnum].set_ylabel("Solidity",weight='bold')
    axes[axesnum].set_xlabel(None)
    axes[axesnum].spines['top'].set_visible(False)
    axes[axesnum].spines['right'].set_visible(False)
    axes[axesnum].spines['left'].set_linewidth(2)
    axes[axesnum].spines['bottom'].set_linewidth(2)
    axes[axesnum].tick_params(axis='both',which='major',width=2,length = 6)
    fig.tight_layout()
    plt.show()
    exit()

    df = df_ori
    df['Eccentricity'] = df['axis_major_length']/df['axis_minor_length']
    df_all=df.copy()
    df_all['Group'] = 'All'
    df_combined = pd.concat([df,df_all],ignore_index=True)
    df = df_combined
    fignum += 1; plt.figure(fignum)
    fig, axes = plt.subplots(2,1,num=fignum)
    fig.suptitle("Nuclear Eccentricity",weight='bold')
    sns.barplot(data=df[df['Date']=='2025-07-20'],x='Group',y='Eccentricity',order=order,estimator=np.mean,palette=palette, errorbar='se',capsize=0.2,ax=axes[0])
    sns.stripplot(data=df[df['Date']=='2025-07-20'],x='Group',y='Eccentricity',order=order,ax=axes[0])
    axesnum = 0
    axes[axesnum].set_title("Exposure 24 hr",weight='bold')
    axes[axesnum].set_ylabel("Solidity",weight='bold')
    axes[axesnum].set_xlabel(None)
    axes[axesnum].spines['top'].set_visible(False)
    axes[axesnum].spines['right'].set_visible(False)
    axes[axesnum].spines['left'].set_linewidth(2)
    axes[axesnum].spines['bottom'].set_linewidth(2)
    sns.barplot(data=df[df['Date']=='2025-07-21'],x='Group',y='Eccentricity',order=order,estimator=np.mean,palette=palette, errorbar='se',capsize=0.2,ax=axes[1])
    sns.stripplot(data=df[df['Date']=='2025-07-21'],x='Group',y='Eccentricity',order=order,ax=axes[1])
    axesnum = 1
    axes[axesnum].set_title("Recover 24 hr",weight='bold')
    axes[axesnum].set_ylabel("Solidity",weight='bold')
    axes[axesnum].set_xlabel(None)
    axes[axesnum].spines['top'].set_visible(False)
    axes[axesnum].spines['right'].set_visible(False)
    axes[axesnum].spines['left'].set_linewidth(2)
    axes[axesnum].spines['bottom'].set_linewidth(2)
    fig.tight_layout()
    plt.show()
    #with tifffile.Tifffile(path + file + '.tif') as tiffimg:
    #    img = tiffimg.asarray()
    #label_masks = np.load(path+file+'_masks.npy')
    #props = regionprops_table(label_image = label_masks, intensity_image = img)

    # if I want to use pandas
    # props_table = regionprops_table(label_image = label_masks, intensity_image = img)
    # tables = pd.DataFrame(props_table)