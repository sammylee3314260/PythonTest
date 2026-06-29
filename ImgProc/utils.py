### Maintain a utility functions library
# import sys
import glob
import os
import re
import numpy as np

def get_filepath(try_gui:bool = True, sys_argv = None, given_path = None):
    """If try_gui, will try to run Windows gui system.\n
       sys_argv should only put in: sys.argv.
       If given_path, will return given_path.
    """
    if sys_argv[1:]:
        if isinstance(sys_argv[1], str): return sys_argv[1]
        else: print(f"sys.argv[1] {sys_argv[1]} is not string, keep going.")
    if given_path: return given_path
    import subprocess
    cmd = [
        "powershell.exe",
        "-NoProfile",
        "-Command",
        "& { $p = New-Object -ComObject Shell.Application; " +
        "$f = $p.BrowseForFolder(0, 'Select a folder', 0); " +
        "if ($f) { $f.Self.Path } }"
    ]
    if try_gui:
        try:
            win_path = subprocess.check_output(cmd).decode('utf-8').strip()
            if win_path:
                unix_path = subprocess.check_output(["wslpath", "-u", win_path]).decode('utf-8').strip()
                return unix_path
        except subprocess.CalledProcessError:
            print("Cannot open windows file explorer. Try cli.")
    
    import readline
    readline.set_completer_delims(' \t\n;')
    readline.parse_and_bind("tab: complete")
    def path_completer(text, state):
        return (glob.glob(os.path.expanduser(text) + '*') + [None])[state]
    readline.set_completer(path_completer)
    parent_folder = input("Please enter the path to the parent folder, you can use tab:\n")
    return parent_folder

def natural_sort_key(s):
    """sort filename (e.g. _2_ < _10_)"""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

def normalize_frame(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8: return frame
    f_min, f_max = frame.min(), frame.max()
    if f_max == f_min:
        return np.zeros_like(frame, dtype=np.uint8)
    norm = (frame.astype(np.float32) - f_min) / (f_max - f_min) * 255
    return norm.astype(np.uint8)