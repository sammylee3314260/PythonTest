import tkinter as tk
from tkinter import filedialog
import tifffile
import numpy as np
import os


root = tk.Tk()
root.withdraw()

folder_path = filedialog.askdirectory(title='Select Folder',initialdir='C:/Users/pathaklab/Box/HyperOsmo/')

if not folder_path:
    print('No folder selected.')
    exit()

file_path = filedialog.askopenfilenames(
    title='Select Files, Cancel to Batch process all qualified files',
    filetypes=(('Czi files', '*.czi'), ('all files', '*.*')),
    initialdir=folder_path
)

if not file_path:
    selected_files = [
        f
        for f in os.listdir(folder_path)
        if f.lower().endswith('.czi')
    ]
    print(f'No files selected. Process these {len(selected_files)} files under folder {folder_path}:\n',selected_files)
else:
    selected_files = []
    folder_path = ''
    for file in file_path:
        spl = os.path.split(file)
        if folder_path == '': folder_path = spl[0]
        selected_files.append(spl[1])
    print(f'Process selected {len(selected_files)} files under folder {folder_path}:\n',selected_files)

#czi 2 tiff
if False:
    from image_utils import cziwr
    cziwr(folder_path,selected_files)

# Tiff 2 jpeg + brigheness & contrast tuning
if False:
    from image_utils import


#tiff2npy
